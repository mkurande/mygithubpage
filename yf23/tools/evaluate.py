"""Register CAD to every reference photo and measure silhouette agreement.

usage: python evaluate.py <params.json> <out_dir> [--init prev_out_dir/cameras.json]

Registration: only camera parameters are optimised (pose within +/-12 deg,
position, focal length / scale). Overall aircraft size is fixed by the
published dimensions, so photo scale is a free camera parameter and
proportions, not absolute size, are what the photos constrain.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(__file__))
import project as P  # noqa: E402
import yf23_model as M  # noqa: E402

REF = os.path.join(os.path.dirname(__file__), "..", "refs")

REFS = {
    "side": dict(photo="sel/side_pav1_inflight.jpg", mask="masks/side_pav1_inflight.png", valid=None,
                 cam=dict(kind="persp", view="side", dist=120.0, free_aspect=True), weight=1.0,
                 note="PAV-1 87-0800 in flight, port side, telephoto. Gear up."),
    "top": dict(photo="sel/top_pav1_inflight.jpg", mask="masks/top_pav1_inflight.png", valid=None,
                cam=dict(kind="persp", view="top", dist=120.0, yaw=90.0, free_aspect=True), weight=1.0,
                note="PAV-1 87-0800 in flight, from above, nose left. Gear up."),
    "top_b": dict(photo="sel/top_holdout_pav1_inflight.jpg", mask="masks/top_holdout_pav1_inflight.png", valid=None,
                  cam=dict(kind="persp", view="top", dist=120.0, yaw=-90.0, free_aspect=True), weight=0.0,
                  note="PAV-1 87-0800 in flight, from above, nose right (hold-out check, not used for fitting)."),
    "front": dict(photo="sel/front_pav1_nmusaf.jpg", mask="masks/front_pav1_nmusaf.png",
                  valid="masks/front_pav1_nmusaf_valid.png",
                  cam=dict(kind="persp", view="front", dist=22.0, dist_starts=[16.0, 22.0, 30.0]), weight=1.0,
                  note="PAV-1 at NMUSAF, head-on, close-range perspective; hand-traced; gear excluded."),
}

WORK_SCALE = float(os.environ.get("YF23_WORK_SCALE", "0.35"))  # registration/refinement resolution
ROT_LIM = 12.0


def load_ref(name):
    r = REFS[name]
    m = cv2.imread(os.path.join(REF, r["mask"]), 0) > 127
    v = cv2.imread(os.path.join(REF, r["valid"]), 0) > 127 if r["valid"] else np.ones_like(m)
    return m, v


def cam_vec_keys(cam):
    return ["yaw", "pitch", "roll", "tx", "ty", "f", "dist", "asp"]


def make_cam(base, vec, W, H):
    c = dict(base)
    for k, x in zip(cam_vec_keys(base), vec):
        c[k] = float(x)
    c.update(W=W, H=H, cx=W / 2, cy=H / 2)
    return c


def init_cam(base, mesh, ref_mask, W, H):
    """Moment-based initial guess: match bounding box size and centroid."""
    c = dict(base)
    c.setdefault("yaw", 0.0)
    c.setdefault("pitch", 0.0)
    c.setdefault("roll", 0.0)
    c.update(tx=0.0, ty=0.0, W=W, H=H, cx=W / 2, cy=H / 2)
    c["f"] = 1.0
    uv, d = P.project(mesh.v, dict(c, f=1.0))
    ys, xs = np.nonzero(ref_mask)
    span_ref = max(np.ptp(xs), np.ptp(ys))
    span_cad = max(np.ptp(uv[:, 0]), np.ptp(uv[:, 1]))
    c["f"] = span_ref / span_cad
    uv, _ = P.project(mesh.v, c)
    # shift via camera-plane translation
    du = xs.mean() - uv[:, 0].mean()
    dv = ys.mean() - uv[:, 1].mean()
    c["tx"] = du * c["dist"] / c["f"]
    c["ty"] = dv * c["dist"] / c["f"]
    return c


def register(mesh, base, ref, valid, init=None, iters=2):
    H, W = ref.shape
    w, h = int(W * WORK_SCALE), int(H * WORK_SCALE)
    r_s = cv2.resize(ref.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    v_s = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    keys = cam_vec_keys(base)
    if init is None and base.get("dist_starts"):
        best_c, best_v = None, -1
        for d0 in base["dist_starts"]:
            c = register(mesh, dict(base, dist=d0, dist_starts=None), ref, valid, None, iters=1)
            cs = dict(c); cs["f"] *= WORK_SCALE; cs.update(W=w, H=h, cx=w / 2, cy=h / 2)
            v = P.iou(P.silhouette(mesh, cs) & v_s, r_s & v_s)
            print(f"   start dist={d0}: IoU {v:.4f}", flush=True)
            if v > best_v:
                best_c, best_v = c, v
        init = best_c
    if init is None:
        c0 = init_cam(base, mesh, r_s, w, h)
    else:
        c0 = dict(init)
        c0["f"] *= WORK_SCALE
        c0.update(W=w, H=h, cx=w / 2, cy=h / 2)
    yaw0, pitch0, roll0 = base.get("yaw", 0.0), base.get("pitch", 0.0), base.get("roll", 0.0)

    # optimise in normalised units: degrees, metres, log focal
    free_asp = base.get("free_aspect", False)

    def unpack(z):
        yaw, pitch, roll, tx, ty, lf, ld = z[:7]
        la = z[7] if free_asp else 0.0
        return [yaw, pitch, roll, tx, ty, float(np.exp(lf)), float(np.exp(ld)), float(np.exp(la))]

    def cost(z):
        vec = unpack(z)
        pen = 0.0
        for a, a0 in zip(vec[:3], (yaw0, pitch0, roll0)):
            pen += max(0.0, abs(a - a0) - ROT_LIM) * 0.05
        pen += max(0.0, abs(np.log(vec[7])) - 0.05) * 5.0
        c = make_cam(base, vec, w, h)
        s = P.silhouette(mesh, c)
        return -(P.iou(s & v_s, r_s & v_s)) + pen

    z = [c0["yaw"], c0["pitch"], c0["roll"], c0["tx"], c0["ty"], np.log(c0["f"]), np.log(c0["dist"])]
    if free_asp:
        z.append(np.log(c0.get("asp", 1.0)))
    z = np.array(z)
    # distance only matters through perspective strength; keep it in a sane band
    best = (cost(z), z)
    for it in range(iters):
        res = minimize(cost, best[1], method="Powell",
                       options=dict(xtol=1e-3, ftol=1e-5, maxfev=1500))
        if res.fun < best[0]:
            best = (res.fun, res.x)
    vec = unpack(best[1])
    cam = make_cam(base, vec, W, H)
    cam["f"] = vec[5] / WORK_SCALE
    cam.pop("dist_starts", None)
    return cam


def px_per_m(cam):
    return cam["f"] / cam["dist"]


def evaluate(params_path, out_dir, init_path=None, density=0.4, only=None):
    os.makedirs(out_dir, exist_ok=True)
    p = M.load_params(params_path)
    mesh = M.build(p, density=density, gear_down=False)
    mesh_reg = M.build(p, density=0.22, gear_down=False)
    inits = json.load(open(init_path)) if init_path and os.path.exists(init_path) else {}
    results, cams = {}, {}
    for name, r in REFS.items():
        if only and name not in only:
            continue
        t0 = time.time()
        ref, valid = load_ref(name)
        cam = register(mesh_reg, r["cam"], ref, valid, init=inits.get(name), iters=1 if name in inits else 2)
        cad = P.silhouette(mesh, cam, ss=2)
        cad_v, ref_v = cad & valid, ref & valid
        k = px_per_m(cam)
        b = P.boundary_metrics(cad_v, ref_v, k)
        miss = (ref_v & ~cad_v).sum() / max(ref_v.sum(), 1)
        extra = (cad_v & ~ref_v).sum() / max(ref_v.sum(), 1)
        results[name] = dict(iou=float(P.iou(cad_v, ref_v)), missing_frac=float(miss), excess_frac=float(extra),
                             boundary_mean_m=b["mean"], boundary_p95_m=b["p95"], hausdorff_m=b["hausdorff"],
                             px_per_m_at_cg=float(k), fit_weight=r["weight"], seconds=round(time.time() - t0, 1),
                             cam={kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in cam.items()})
        cams[name] = cam
        img = cv2.imread(os.path.join(REF, r["photo"]))
        ov = P.overlay(img, cad_v, ref_v)
        if r["valid"]:
            cs, _ = cv2.findContours(valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, cs, -1, (255, 0, 255), 2)
        cv2.imwrite(os.path.join(out_dir, f"overlay_{name}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 88])
        np.save(os.path.join(out_dir, f"cad_{name}.npy"), np.packbits(cad_v))
        print(f"{name:6s} IoU={results[name]['iou']:.4f} miss={miss:.3f} extra={extra:.3f} "
              f"bnd_mean={b['mean']:.3f}m p95={b['p95']:.3f}m ({results[name]['seconds']}s)", flush=True)
    fitted = [n for n in results if REFS[n]["weight"] > 0]
    summary = dict(params=os.path.relpath(params_path, os.path.join(out_dir, "..")),
                   mean_iou_fitted=float(np.mean([results[n]["iou"] for n in fitted])) if fitted else None,
                   results=results, triangles_eval=int(len(mesh.f)))
    json.dump(summary, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    json.dump(cams, open(os.path.join(out_dir, "cameras.json"), "w"), indent=1)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("params")
    ap.add_argument("out")
    ap.add_argument("--init")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    s = evaluate(a.params, a.out, a.init, only=a.only)
    print("mean IoU (fitted refs):", s["mean_iou_fitted"])
