"""Quick z-buffer shaded renders (numba) for visual regression checks."""
import sys
import math
import cv2
import numpy as np
from numba import njit
import yf23_model as M
import project as P


@njit(cache=True)
def _zbuf(tri, z, shade, H, W):
    img = np.full((H, W), 255.0)
    zb = np.full((H, W), 1e9)
    for t in range(tri.shape[0]):
        x0, y0, x1, y1, x2, y2 = tri[t, 0, 0], tri[t, 0, 1], tri[t, 1, 0], tri[t, 1, 1], tri[t, 2, 0], tri[t, 2, 1]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0.0:
            continue
        xmin = max(int(min(x0, min(x1, x2))), 0); xmax = min(int(max(x0, max(x1, x2))) + 1, W - 1)
        ymin = max(int(min(y0, min(y1, y2))), 0); ymax = min(int(max(y0, max(y1, y2))) + 1, H - 1)
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                px, py = x + 0.5, y + 0.5
                w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
                w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
                w2 = 1 - w0 - w1
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    d = w0 * z[t, 0] + w1 * z[t, 1] + w2 * z[t, 2]
                    if d < zb[y, x]:
                        zb[y, x] = d
                        img[y, x] = shade[t]
    return img


def lookat_project(V, eye, target, f, W, H):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    Q = V - eye
    d = Q @ fwd
    uv = np.stack([W / 2 + f * (Q @ right) / d, H / 2 - f * (Q @ up) / d], -1)
    return uv, d


def render(mesh, cam, light=(0.4, -0.5, 0.75)):
    if cam.get("kind") == "lookat":
        uv, d = lookat_project(mesh.v, cam["eye"], cam["target"], cam["f"], cam["W"], cam["H"])
    else:
        uv, d = P.project(mesh.v, cam)
    V = mesh.v[mesh.f]
    n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    s = 70 + 170 * np.abs(n @ L)
    keep = (d[mesh.f] > 0.1).all(1)  # near-plane cull: drop triangles touching the region behind the camera
    img = _zbuf(np.ascontiguousarray(uv[mesh.f][keep]), np.ascontiguousarray(d[mesh.f][keep]), s[keep], cam["H"], cam["W"])
    return img.astype(np.uint8)


def views(params_path, out, gear=True):
    p = M.load_params(params_path)
    m = M.build(p, 0.7, gear_down=gear)
    W, H = 900, 520
    T = [10.3, 0, 2.2]
    mk = lambda eye, f=1150: dict(kind="lookat", eye=eye, target=T, f=f, W=W, H=H)
    cams = [mk([-8, -22, 12]),   # front-left, above
            mk([30, -20, 11]),   # rear-left, above
            mk([-14, 14, 3.0]),  # front-right, low
            mk([24, 16, -5])]    # rear-right, below
    ims = [render(m, c) for c in cams]
    g = np.vstack([np.hstack(ims[:2]), np.hstack(ims[2:])])
    cv2.imwrite(out, g)
    return out


if __name__ == "__main__":
    print(views(sys.argv[1], sys.argv[2]))
