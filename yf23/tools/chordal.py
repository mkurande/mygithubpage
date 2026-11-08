"""Fuselage chordal error: normal deviation between mesh facets and the true parametric surface at facet centres."""
import copy
import numpy as np
import yf23_model as M


def _surf(p, x, t):
    g = lambda k: float(M.curve(p, k, [x])[0])
    zc = g("fus_chine_z")
    return M.chined_section(max(g("fus_halfwidth"), 1e-3), zc, max(g("fus_top"), zc + 1e-3), min(g("fus_bottom"), zc - 1e-3),
                            g("fus_exp_upper"), g("fus_exp_lower"), len(t), kchine=p.get("chine_sharp", 0.0), t=t)


def fuselage_chordal(p, density, adaptive):
    x0, x1 = p["fus_x"]
    nx, nu = int(round(400 * density)), int(round(160 * density))
    if adaptive:
        xs = M.adaptive_stations(p, ["fus_halfwidth", "fus_top", "fus_bottom", "fus_chine_z"], x0, x1, nx,
                                 extra=[(p["can_x"][0], 2.0), (p["can_x"][1], 1.0), (0.0, 4.0)])
        tt = M.chine_clustered(nu)
    else:
        s = np.linspace(0, 1, nx)
        xs = x0 + (x1 - x0) * (0.5 - 0.5 * np.cos(np.pi * s))
        tt = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    tt2 = np.append(tt, 2 * np.pi)
    tm = 0.5 * (tt2[:-1] + tt2[1:])
    out = []
    for i in range(nx - 1):
        A, B = _surf(p, xs[i], tt2), _surf(p, xs[i + 1], tt2)
        true = _surf(p, 0.5 * (xs[i] + xs[i + 1]), tm)
        A3 = np.c_[np.full(len(A), xs[i]), A]
        B3 = np.c_[np.full(len(B), xs[i + 1]), B]
        T3 = np.c_[np.full(len(true), 0.5 * (xs[i] + xs[i + 1])), true]
        c = 0.25 * (A3[:-1] + A3[1:] + B3[:-1] + B3[1:])
        n = np.cross(B3[1:] - A3[:-1], A3[1:] - B3[:-1])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        out.append(np.abs(np.sum((T3 - c) * n, axis=1)))
    return np.concatenate(out), 2 * (nx - 1) * nu


if __name__ == "__main__":
    import sys
    p = M.load_params(sys.argv[1])
    for label, d, a in (("uniform d=1.0 (iter07 export)", 1.0, False), ("adaptive d=2.057 (5x export)", 2.057, True)):
        e, n = fuselage_chordal(p, d, a)
        print(f"{label}: {n:,} fuselage triangles, normal chordal error mean {e.mean()*1000:.3f} mm, p99 {np.percentile(e,99)*1000:.2f} mm, max {e.max()*1000:.2f} mm")
