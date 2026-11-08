"""Iteration with the photogrammetry term: silhouette IoU (side/top/front) + SfM point distances.

usage: YF23_WORK_SCALE=0.5 python run_iteration_pts.py <n> [--w 0.3] [--sweeps 3]
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import evaluate as E, panel as PN, refine as R, run_iteration as RI
import points_term as PT, yf23_model as M

ROOT = os.path.join(os.path.dirname(__file__), "..")
SFM = os.path.join(ROOT, "sfm", "nmusaf")


def point_metrics(p, al, Ps, seed=0):
    Q = PT.to_cad(Ps, al)
    fit, hold = PT.select_aircraft(Q, p, band=0.35, seed=seed)
    m = M.build(p, 0.4, gear_down=False)
    out = {}
    for k, q in (("fit", fit), ("holdout", hold)):
        d = PT.signed_distance(m, q)
        out[k] = dict(n=int(len(q)), median_abs_cm=round(float(np.median(np.abs(d))) * 100, 2),
                      mean_abs_cm=round(float(np.mean(np.abs(d))) * 100, 2), p90_abs_cm=round(float(np.percentile(np.abs(d), 90)) * 100, 2),
                      median_signed_cm=round(float(np.median(d)) * 100, 2))
    return out, fit, hold


def run(n, w=0.3, sweeps=3, from_params=None):
    prev, cur = f"iter_{n - 1:02d}", f"iter_{n:02d}"
    p = M.load_params(from_params or os.path.join(ROOT, "cad", "params", prev + ".json"))
    cams = json.load(open(os.path.join(ROOT, "iterations", prev, "cameras.json")))
    Ps = PT.load_points(SFM)
    alp = os.path.join(SFM, f"align_{prev}.json")
    al0 = json.load(open(alp if os.path.exists(alp) else os.path.join(SFM, "align_icp.json")))
    al, ninl = PT.realign(p, Ps, al0)
    before, fit, hold = point_metrics(p, al, Ps)
    print(f"[{cur}] before: points fit {before['fit']}  holdout {before['holdout']}", flush=True)
    p2, f = R.refine(p, cams, sweeps=sweeps, log=lambda s: print(s, flush=True), points=fit, w_points=w)
    p2["_note"] = (f"iteration {n}: refinement with silhouette IoU (side/top/front) plus SfM point-to-surface term "
                   f"(w={w}, {len(fit)} fit points from {os.path.relpath(SFM, ROOT)}), cameras from {prev}; then re-registered.")
    pp = os.path.join(ROOT, "cad", "params", cur + ".json")
    json.dump(p2, open(pp, "w"), indent=1)
    al2, _ = PT.realign(p2, Ps, al)
    json.dump(al2, open(os.path.join(SFM, f"align_{cur}.json"), "w"), indent=1)
    after, _, _ = point_metrics(p2, al2, Ps)
    print(f"[{cur}] after: points fit {after['fit']}  holdout {after['holdout']}", flush=True)
    out = os.path.join(ROOT, "iterations", cur)
    s = E.evaluate(pp, out, os.path.join(ROOT, "iterations", prev, "cameras.json"))
    s["sfm_points"] = dict(before=before, after=after, icp_inliers=ninl, scale_sfm_per_m=al2["s"])
    json.dump(s, open(os.path.join(out, "metrics.json"), "w"), indent=1)
    PN.panel(out)
    RI.history_append(n, s, p2["_note"])
    h = json.load(open(os.path.join(ROOT, "iterations", "history.json")))
    for e in h:
        if e["iteration"] == n:
            e["sfm_points_after"] = after
            e["sfm_points_before"] = before
    json.dump(h, open(os.path.join(ROOT, "iterations", "history.json"), "w"), indent=1)
    print(f"[{cur}] mean IoU {s['mean_iou_fitted']:.4f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--w", type=float, default=0.3)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--from-params")
    a = ap.parse_args()
    run(a.n, a.w, a.sweeps, a.from_params)
