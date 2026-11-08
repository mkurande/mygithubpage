"""One refinement iteration: refine geometry -> re-register cameras -> evaluate -> panel -> history.

usage: python run_iteration.py <n> [--sweeps 3] [--groups ...]
Reads iter_{n-1}; writes cad/params/iter_{n}.json and iterations/iter_{n}/.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import evaluate as E  # noqa: E402
import panel as PN  # noqa: E402
import refine as R  # noqa: E402
import yf23_model as M  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def history_append(n, summary, note):
    hp = os.path.join(ROOT, "iterations", "history.json")
    h = json.load(open(hp)) if os.path.exists(hp) else []
    h = [e for e in h if e["iteration"] != n]
    h.append(dict(iteration=n, note=note, mean_iou_fitted=summary["mean_iou_fitted"],
                  iou={k: round(v["iou"], 4) for k, v in summary["results"].items()},
                  boundary_mean_cm={k: round(v["boundary_mean_m"] * 100, 1) for k, v in summary["results"].items()},
                  boundary_p95_cm={k: round(v["boundary_p95_m"] * 100, 1) for k, v in summary["results"].items()},
                  aspect={k: round(v["cam"].get("asp", 1.0), 4) for k, v in summary["results"].items()}))
    h.sort(key=lambda e: e["iteration"])
    json.dump(h, open(hp, "w"), indent=1)


def run(n, sweeps=3, groups=None, note=None, from_params=None):
    prev = f"iter_{n - 1:02d}"
    cur = f"iter_{n:02d}"
    p = M.load_params(from_params or os.path.join(ROOT, "cad", "params", prev + ".json"))
    cams = json.load(open(os.path.join(ROOT, "iterations", prev, "cameras.json")))
    print(f"[{cur}] refine from {prev} (groups={groups or 'all'})", flush=True)
    p2, f = R.refine(p, cams, sweeps=sweeps, groups=groups, log=lambda s: print(s, flush=True))
    p2["_note"] = note or f"iteration {n}: pattern-search refinement of {prev} with {prev} cameras fixed (objective {f:.5f}), then cameras re-registered."
    pp = os.path.join(ROOT, "cad", "params", cur + ".json")
    json.dump(p2, open(pp, "w"), indent=1)
    out = os.path.join(ROOT, "iterations", cur)
    s = E.evaluate(pp, out, os.path.join(ROOT, "iterations", prev, "cameras.json"))
    PN.panel(out)
    history_append(n, s, p2["_note"])
    print(f"[{cur}] mean IoU {s['mean_iou_fitted']:.4f}", flush=True)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("n", type=int)
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--groups", nargs="*")
    ap.add_argument("--from-params")
    a = ap.parse_args()
    run(a.n, a.sweeps, a.groups, from_params=a.from_params)
