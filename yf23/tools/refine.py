"""CAD-refinement step: constrained pattern search on shape parameters.

usage: python refine.py <in_params> <out_params> <cameras.json> [--sweeps N] [--groups g1 g2 ...]

Cameras are held fixed (they come from the previous registration); the
objective is the mean silhouette IoU over the fitted references (side, top,
front) at registration resolution, minus a smoothness penalty on every curve
table. Locked: nose at x=0, fuselage end, wing semispan (published span),
and overall length (penalised if the aftmost point leaves 20.55 m +/- 2 cm).
The hold-out top photo is never used here.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import evaluate as E  # noqa: E402
import project as P  # noqa: E402
import yf23_model as M  # noqa: E402

L_TOTAL = 20.55

# (path, step) — path is a tuple into the params dict; tables use (key, i, 1)
TABLE_STEPS = {
    "fus_halfwidth": 0.06, "fus_top": 0.05, "fus_bottom": 0.05, "fus_chine_z": 0.04,
    "fus_exp_upper": 0.25, "fus_exp_lower": 0.25,
    "can_halfwidth": 0.04, "can_top": 0.04,
    "nac_y": 0.06, "nac_halfwidth": 0.05, "nac_top": 0.04, "nac_bottom": 0.04,
    "nac_bottom_ratio": 0.06, "nac_bottom_shift": 0.05, "nac_exp": 0.5,
}
SCALAR_STEPS = {
    ("wing", "le_sweep_deg"): 0.5, ("wing", "te_sweep_deg"): 0.5, ("wing", "x_le_c"): 0.08, ("wing", "x_te_c"): 0.08,
    ("wing", "z_root"): 0.04, ("wing", "dihedral_deg"): 0.5, ("wing", "tc_root"): 0.005,
    ("vtail", "x_le_root"): 0.08, ("vtail", "root_chord"): 0.08, ("vtail", "le_sweep_deg"): 1.0, ("vtail", "span"): 0.08,
    ("vtail", "tip_chord"): 0.06, ("vtail", "kink_frac"): 0.03, ("vtail", "kink_x"): 0.08,
    ("vtail", "cant_from_vertical_deg"): 1.0, ("vtail", "y_root"): 0.05, ("vtail", "z_root"): 0.05,
    ("can_exp",): 0.2, ("chine_sharp",): 0.1,
    ("nacelle", "x", 0): 0.08,
}
GROUPS = {
    "fuselage": ["fus_"], "canopy": ["can_"], "nacelle": ["nac_", "nacelle"], "wing": ["wing"], "vtail": ["vtail"],
    "exhaust": ["exhaust"], "misc": ["chine_sharp"],
}


def get(p, path):
    if path[0] == "__shift__":
        return 0.0
    for k in path:
        p = p[k]
    return p


def put(p, path, v):
    if path[0] == "__shift__":  # shift the whole table by v (relative move)
        for row in p[path[1]]:
            row[1] += v
        return
    for k in path[:-1]:
        p = p[k]
    p[path[-1]] = v


def coords(p, groups):
    out = []
    for key, st in TABLE_STEPS.items():
        if key not in p or not any(key.startswith(g) for gr in groups for g in GROUPS[gr]):
            continue
        for i in range(len(p[key])):
            out.append(((key, i, 1), st))
        out.append((("__shift__", key), st))
    for path, st in SCALAR_STEPS.items():
        if not any(path[0].startswith(g) for gr in groups for g in GROUPS[gr]):
            continue
        try:
            get(p, path)
        except (KeyError, IndexError, TypeError):
            continue
        out.append((path, st))
    if "exhaust" in groups and "exhaust" in p:
        for j in range(len(p["exhaust"]["walls"])):
            for k, st in (("x_tip", 0.06), ("y", 0.05), ("x0", 0.1), ("z_bot", 0.05), ("tip_dy", 0.05)):
                out.append((("exhaust", "walls", j, k), st))
    return out


def smooth_pen(p):
    pen = 0.0
    for key in TABLE_STEPS:
        if key not in p or len(p[key]) < 3:
            continue
        t = np.asarray(p[key], float)
        x, y = t[:, 0], t[:, 1]
        sl = np.diff(y) / np.diff(x)
        pen += np.sum(np.diff(sl) ** 2) * 0.002
    return pen


class Objective:
    def __init__(self, cams, names=("side", "top", "front")):
        self.items = []
        for n in names:
            ref, valid = E.load_ref(n)
            c = dict(cams[n])
            H, W = ref.shape
            w, h = int(W * E.WORK_SCALE), int(H * E.WORK_SCALE)
            c["f"] *= E.WORK_SCALE
            c.update(W=w, H=h, cx=w / 2, cy=h / 2)
            r = cv2.resize(ref.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            v = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            self.items.append((n, c, r & v, v, E.REFS[n]["weight"]))
        self.evals = 0

    def ious(self, p):
        mesh = M.build(p, density=0.22, gear_down=False)
        out = {}
        for n, c, r, v, w in self.items:
            out[n] = P.iou(P.silhouette(mesh, c) & v, r)
        return out, mesh

    def __call__(self, p):
        self.evals += 1
        try:
            d, mesh = self.ious(p)
        except Exception:
            return -1e9
        wsum = sum(w for *_, w in self.items)
        score = sum(d[n] * w for n, _, _, _, w in self.items) / wsum
        xmax = mesh.v[:, 0].max()
        pen = max(0.0, abs(xmax - L_TOTAL) - 0.02) * 0.5 + smooth_pen(p)
        return score - pen


def refine(p, cams, sweeps=3, groups=None, log=print, time_budget=None):
    groups = groups or list(GROUPS)
    obj = Objective(cams)
    cs = coords(p, groups)
    steps = {i: st for i, (_, st) in enumerate(cs)}
    best = obj(p)
    log(f"  start objective {best:.5f}  ({len(cs)} coordinates)")
    t0 = time.time()
    for s in range(sweeps):
        improved = 0
        for i, (path, _) in enumerate(cs):
            st = steps[i]
            v0 = get(p, path)
            moved = False
            for sgn in (1, -1):
                q = copy.deepcopy(p)
                put(q, path, v0 + sgn * st)
                f = obj(q)
                if f > best + 1e-6:
                    best, p, moved = f, q, True
                    # keep going in the same direction while it helps
                    while True:
                        q = copy.deepcopy(p)
                        put(q, path, get(p, path) + sgn * st)
                        f = obj(q)
                        if f > best + 1e-6:
                            best, p = f, q
                        else:
                            break
                    break
            if moved:
                improved += 1
            else:
                steps[i] = st * 0.5
            if time_budget and time.time() - t0 > time_budget:
                break
        log(f"  sweep {s + 1}: objective {best:.5f}, {improved}/{len(cs)} coordinates moved, {obj.evals} evals, {time.time() - t0:.0f}s")
        if time_budget and time.time() - t0 > time_budget:
            break
    return p, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("cams")
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    p = M.load_params(a.inp)
    cams = json.load(open(a.cams))
    p2, f = refine(p, cams, a.sweeps, a.groups)
    p2["_note"] = a.note or f"refined from {os.path.basename(a.inp)} by pattern search (objective {f:.5f})"
    json.dump(p2, open(a.out, "w"), indent=1)
