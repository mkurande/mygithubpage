"""Multi-view silhouettes from SfM-posed museum photos.

For every registered image: project the CAD (gear down) through the COLMAP camera
(SIMPLE_RADIAL), prompt SAM with the projected box plus points inside / just
outside the projection, keep SAM's mask (it follows the real image edges), and
score IoU(CAD, SAM). Masks are cached so later iterations only re-project.
"""
import json, os, sys
import cv2
import numpy as np
import pycolmap

sys.path.insert(0, os.path.dirname(__file__))
import project as P
import yf23_model as M

ROOT = os.path.join(os.path.dirname(__file__), "..")


class View:
    def __init__(self, im, cam):
        self.name = im.name
        self.Rc = im.cam_from_world().rotation.matrix()
        self.tc = im.cam_from_world().translation
        f, cx, cy, k = cam.params
        self.f, self.cx, self.cy, self.k = f, cx, cy, k
        self.W, self.H = cam.width, cam.height


def load_views(workdir):
    r = pycolmap.Reconstruction(os.path.join(workdir, "sparse", "0"))
    return r, [View(im, r.cameras[im.camera_id]) for im in r.images.values()]


def project_cad(V, view, al):
    s, R, t = al["s"], np.array(al["R"]), np.array(al["t"])
    Xw = s * (V @ R.T) + t
    Xc = Xw @ view.Rc.T + view.tc
    z = Xc[:, 2]
    x, y = Xc[:, 0] / z, Xc[:, 1] / z
    r2 = x * x + y * y
    d = 1 + view.k * r2
    return np.stack([view.cx + view.f * x * d, view.cy + view.f * y * d], -1), z


def silhouette(mesh, view, al, scale=1.0):
    uv, z = project_cad(mesh.v, view, al)
    keep = (z[mesh.f] > 0.2).all(1)
    tri = np.ascontiguousarray(uv[mesh.f][keep] * scale, dtype=np.float64)
    H, W = int(view.H * scale), int(view.W * scale)
    return P._raster(tri, H, W).astype(bool)


def sam_mask(img, cad, rng):
    """SAM on a crop around the projected CAD; of SAM's three edge-following candidates keep the one
    that best agrees with the projection (rejects background grabs without copying the CAD outline)."""
    import sam_seg as SS
    ys, xs = np.nonzero(cad)
    if len(xs) < 200:
        return None, 0.0
    H, W = cad.shape
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    span = max(x1 - x0, y1 - y0)
    cx0, cy0 = int(max(0, x0 - 0.15 * span)), int(max(0, y0 - 0.15 * span))
    cx1, cy1 = int(min(W, x1 + 0.15 * span)), int(min(H, y1 + 0.15 * span))
    crop, cc = img[cy0:cy1, cx0:cx1], cad[cy0:cy1, cx0:cx1]
    pad = 0.03 * span
    box = [max(0, x0 - pad - cx0), max(0, y0 - pad - cy0), min(cx1 - cx0 - 1, x1 + pad - cx0), min(cy1 - cy0 - 1, y1 + pad - cy0)]
    sz = max(3, int(0.035 * span))
    inner = cv2.erode(cc.astype(np.uint8), np.ones((sz, sz), np.uint8)) > 0
    far = (cv2.dilate(cc.astype(np.uint8), np.ones((4 * sz, 4 * sz), np.uint8)) > 0) & ~(cv2.dilate(cc.astype(np.uint8), np.ones((2 * sz, 2 * sz), np.uint8)) > 0)
    def pick(mask, n):
        yy, xx = np.nonzero(mask)
        if len(xx) == 0:
            return []
        i = rng.choice(len(xx), min(n, len(xx)), replace=False)
        return list(zip(xx[i], yy[i]))
    ms, scs = SS.segment_all(crop, box=box, pos=pick(inner, 12), neg=pick(far, 12))
    k = int(np.argmax([P.iou(m, cc) for m in ms]))
    full = np.zeros_like(cad)
    full[cy0:cy1, cx0:cx1] = ms[k]
    return full, float(scs[k])


def build_masks(workdir, params, al, out_dir, names=None):
    os.makedirs(out_dir, exist_ok=True)
    r, views = load_views(workdir)
    p = M.load_params(params)
    mesh = M.build(p, 0.5, gear_down=True)
    rng = np.random.default_rng(0)
    rows = []
    for v in views:
        if names and v.name not in names:
            continue
        img = cv2.imread(os.path.join(workdir, "images", v.name))
        cad = silhouette(mesh, v, al)
        mp = os.path.join(out_dir, v.name[:-4] + ".png")
        if os.path.exists(mp):
            m = cv2.imread(mp, 0) > 127; sc = float("nan")
        else:
            m, sc = sam_mask(img, cad, rng)
            if m is None:
                continue
            cv2.imwrite(mp, (m * 255).astype(np.uint8))
        inside = cad.sum() / max(cad.size, 1)
        rows.append(dict(name=v.name, sam_score=sc, iou=float(P.iou(cad, m)), cad_frac=float(inside),
                         truncated=bool(cad[0].any() or cad[-1].any() or cad[:, 0].any() or cad[:, -1].any())))
        print(f"{v.name:40s} IoU {rows[-1]['iou']:.3f} sam {sc:.3f}", flush=True)
    json.dump(rows, open(os.path.join(out_dir, "index.json"), "w"), indent=1)
    return rows


if __name__ == "__main__":
    al = json.load(open(sys.argv[3]))
    build_masks(sys.argv[1], sys.argv[2], al, sys.argv[4])


def band_of(cad, frac=0.035):
    ys, xs = np.nonzero(cad)
    if len(xs) == 0:
        return np.zeros_like(cad)
    span = max(xs.ptp() if hasattr(xs, "ptp") else np.ptp(xs), np.ptp(ys))
    k = max(3, int(frac * span)) | 1
    return cv2.dilate(cad.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def banded_iou(cad, sam, band):
    return float(P.iou(cad & band, sam & band))


def score_views(workdir, params, al, mask_dir, names=None, frac=0.035, band_params=None):
    """Banded IoU per view. The band comes from `band_params` (default: the scored params) so
    different revisions can be compared on an identical evaluation region."""
    r, views = load_views(workdir)
    mesh = M.build(M.load_params(params), 0.5, gear_down=True)
    bmesh = M.build(M.load_params(band_params), 0.5, gear_down=True) if band_params else mesh
    out = {}
    for v in views:
        if names and v.name not in names:
            continue
        mp = os.path.join(mask_dir, v.name[:-4] + ".png")
        if not os.path.exists(mp):
            continue
        sam = cv2.imread(mp, 0) > 127
        cad = silhouette(mesh, v, al)
        band = band_of(silhouette(bmesh, v, al), frac) & ~gear_zone(bmesh, v, al)
        out[v.name] = banded_iou(cad, sam, band)
    return out


def gear_zone(mesh, view, al):
    """Dilated projection of the landing gear: excluded from comparisons (SAM drops thin struts)."""
    g = mesh.part == M.PARTS.index("gear")
    gm = M.Mesh(mesh.v, mesh.f[g], mesh.part[g])
    sil = silhouette(gm, view, al)
    ys, xs = np.nonzero(sil)
    if len(xs) == 0:
        return sil
    k = max(5, int(0.02 * max(view.W, view.H))) | 1
    return cv2.dilate(sil.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def prepare_terms(workdir, p, al, mask_dir, names, scale=0.35, frac=0.035):
    r, views = load_views(workdir)
    mesh = M.build(p, 0.5, gear_down=True)
    terms = []
    for v in views:
        if v.name not in names:
            continue
        sam = cv2.imread(os.path.join(mask_dir, v.name[:-4] + ".png"), 0) > 127
        cad = silhouette(mesh, v, al)
        band = band_of(cad, frac) & ~gear_zone(mesh, v, al)
        W, H = int(v.W * scale), int(v.H * scale)
        sm = cv2.resize(sam.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        bd = cv2.resize(band.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        terms.append((v, al, sm, bd, scale))
    return terms
