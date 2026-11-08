"""Detail layer: canopy frame, air-data probes and articulated landing gear.

Enabled when the params carry a "detail" block. Evidence: NMUSAF walk-around
photos of PAV-1 (refs/raw/United_States_Air_Force_*), which show the windscreen
bow and canopy sill, an L-shaped probe on each lower forebody side, a single
nose wheel on an oleo strut with a torque link and one large door, and single
main wheels on short struts under the nacelles.
"""
import math

import numpy as np

import yf23_model as M


def _frames(path):
    """Rotation-minimising frames along a polyline."""
    T = np.gradient(path, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    a = np.array([0, 0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0, 0])
    N = [np.cross(T[0], a)]
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, len(path)):
        n = N[-1] - T[i] * (N[-1] @ T[i])
        N.append(n / (np.linalg.norm(n) + 1e-12))
    N = np.array(N)
    B = np.cross(T, N)
    return N, B


def tube(path, r, m, part, caps=True, prof=None):
    """Sweep a circle (or a closed 2-D profile in the N-B plane) along a path."""
    path = np.asarray(path, float)
    N, B = _frames(path)
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    pr = prof if prof is not None else np.stack([np.cos(t), np.sin(t)], -1)
    rr = np.broadcast_to(np.asarray(r, float), (len(path),))
    P = path[:, None, :] + rr[:, None, None] * (pr[None, :, 0, None] * N[:, None, :] + pr[None, :, 1, None] * B[:, None, :])
    return M.grid_mesh(P, part, close_u=True, cap_start=caps, cap_end=caps)


def torus_y(c, R, tube_r, half_w, nu, nv, part):
    """Tyre: revolution about the y axis of a rounded-rectangle profile."""
    t = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    # profile in (radial, axial) — superellipse gives a tyre-like flat tread
    a = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    e = 2 / 3.0
    pr_r = (R - tube_r) + tube_r * np.sign(np.cos(a)) * np.abs(np.cos(a)) ** e
    pr_a = half_w * np.sign(np.sin(a)) * np.abs(np.sin(a)) ** e
    P = np.zeros((nv, nu, 3))
    for i, th in enumerate(t):
        P[i, :, 0] = c[0] + pr_r * math.cos(th)
        P[i, :, 2] = c[2] + pr_r * math.sin(th)
        P[i, :, 1] = c[1] + pr_a
    P = np.concatenate([P, P[:1]], 0)
    return M.grid_mesh(P, part, close_u=True)


def box(c, size, part):
    c, s = np.asarray(c, float), np.asarray(size, float) / 2
    P = np.array([[[c[0] - s[0], c[1] - s[1], c[2] - s[2]], [c[0] - s[0], c[1] + s[1], c[2] - s[2]],
                   [c[0] - s[0], c[1] + s[1], c[2] + s[2]], [c[0] - s[0], c[1] - s[1], c[2] + s[2]]],
                  [[c[0] + s[0], c[1] - s[1], c[2] - s[2]], [c[0] + s[0], c[1] + s[1], c[2] - s[2]],
                   [c[0] + s[0], c[1] + s[1], c[2] + s[2]], [c[0] + s[0], c[1] - s[1], c[2] + s[2]]]])
    return M.grid_mesh(P, part, close_u=True, cap_start=True, cap_end=True)


def plate(corners, thick, part):
    """Thin plate from 4 corners (ordered), extruded along its normal."""
    C = np.asarray(corners, float)
    n = np.cross(C[1] - C[0], C[3] - C[0])
    n /= np.linalg.norm(n)
    P = np.stack([C - n * thick / 2, C + n * thick / 2], 0)
    return M.grid_mesh(P, part, close_u=True, cap_start=True, cap_end=True)


def canopy_frame(p, density):
    D = p["detail"]["canopy_frame"]
    m = max(8, int(12 * density))
    x0, x1 = p["can_x"]
    xs = np.linspace(x0 + 0.05, x1 - 0.05, max(20, int(120 * density)))
    w = np.maximum(M.curve(p, "can_halfwidth", xs), 1e-3)
    zb = M.fus_surface_z(p, xs, w)
    out = []
    for sgn in (1, -1):
        out.append(tube(np.stack([xs, sgn * w, zb], -1), D["sill_r"], m, "frame"))
    # windscreen bow: follows the canopy section at x_bow
    xb = D["bow_x"]
    wb = float(M.curve(p, "can_halfwidth", [xb])[0])
    zt = float(M.curve(p, "can_top", [xb])[0])
    e = 2 / p["can_exp"]
    t = np.linspace(0.0, np.pi, max(24, int(80 * density)))
    c, s = np.cos(t), np.sin(t)
    yy = wb * np.sign(c) * np.abs(c) ** e
    base = M.fus_surface_z(p, np.full_like(yy, xb), yy) - 0.02
    zz = base + max(zt - float(M.fus_surface_z(p, [xb], [0.0])[0]) + 0.02, 0) * np.abs(s) ** e
    bow = np.stack([np.full_like(t, xb), yy * 1.01, zz + 0.01], -1)
    out.append(tube(bow, D["bow_r"], m, "frame"))
    return out


def probes(p, density):
    D = p["detail"]["probes"]
    m = max(6, int(8 * density))
    out = []
    x = D["x"]
    w = float(M.curve(p, "fus_halfwidth", [x])[0])
    zc = float(M.curve(p, "fus_chine_z", [x])[0])
    zb = float(M.curve(p, "fus_bottom", [x])[0])
    el = float(M.curve(p, "fus_exp_lower", [x])[0])
    # seat the probe base on the lower forebody surface at height zc - dz
    sn = min(D["dz"] / max(zc - zb, 1e-3), 0.999) ** (el / 2)
    y_surf = w * np.sqrt(1 - sn ** 2) ** (2 / el)
    for sgn in (1, -1):
        y0 = sgn * (y_surf - 0.005)
        z0 = zc - D["dz"]
        stem = np.array([[x, y0, z0], [x, y0 + sgn * D["stem"], z0 - 0.01]])
        stem = np.linspace(stem[0], stem[1], 6)
        vane = np.linspace([x, y0 + sgn * D["stem"], z0 - 0.01], [x - D["vane"], y0 + sgn * D["stem"], z0 - 0.01], 8)
        out.append(tube(stem, D["r"], m, "probe"))
        out.append(tube(vane, D["r"] * 0.8, m, "probe"))
    return out


def gear_detailed(p, density):
    G = p["gear"]
    m = max(10, int(20 * density))
    out = []
    for g in G["legs"]:
        main = g["y"] > 0
        for sgn in ((1, -1) if main else (1,)):
            y = sgn * g["y"]
            R, hw = g["wheel_r"], g["wheel_w"] / 2
            wc = np.array([g["x"], y, R])
            # strut sits beside the wheel (inboard for mains, a fork for the nose wheel)
            ys = y - sgn * (hw + 0.07) if main else y
            z_top, z_axle = g["z_top"], R + 0.02
            z_mid = z_axle + 0.45 * (z_top - z_axle)
            out.append(tube(np.linspace([g["x"], ys, z_top], [g["x"], ys, z_mid], 8), g["strut_r"], m, "gear"))
            out.append(tube(np.linspace([g["x"], ys, z_mid + 0.05], [g["x"], ys, z_axle], 8), g["strut_r"] * 0.65, m, "gear"))
            if main:
                out.append(tube(np.linspace([g["x"], ys, z_axle], [g["x"], y, z_axle], 6), g["strut_r"] * 0.5, m, "gear"))
            else:
                for fy in (-1, 1):  # fork legs
                    out.append(tube(np.linspace([g["x"], ys + fy * (hw + 0.04), z_mid], [g["x"], ys + fy * (hw + 0.04), z_axle], 6),
                                    g["strut_r"] * 0.4, m, "gear"))
                out.append(box([g["x"], ys, z_mid], [0.12, 2 * hw + 0.12, 0.06], "gear"))
            # torque link (two arms ahead of the strut)
            zl = 0.5 * (z_mid + z_axle)
            for z0, z1 in ((z_mid, zl), (zl, z_axle + 0.05)):
                out.append(tube(np.linspace([g["x"] - 0.02, ys, z0], [g["x"] - 0.12, ys, z1], 4), 0.02, m // 2, "gear"))
            # drag brace from strut middle forward/up to the bay
            out.append(tube(np.linspace([g["x"], ys, z_mid + 0.1], [g["x"] - 0.6 if not main else g["x"] + 0.6, ys, z_top], 8),
                            g["strut_r"] * 0.35, m, "gear"))
            # tyre and hub
            out.append(torus_y(wc, R, R * 0.32, hw, m, max(24, int(64 * density)), "gear"))
            out.append(tube(np.linspace(wc - [0, hw * 0.8, 0], wc + [0, hw * 0.8, 0], 3), R * 0.62, max(16, int(32 * density)), "gear"))
            # gear door(s)
            if g.get("door_h", 0) > 0:
                d, L = g["door_h"], g["door_len"]
                yo = y + sgn * g.get("door_dy", 0.25) if main else y - g.get("door_dy", 0.25)
                x0, x1 = g["x"] - L / 2, g["x"] + L / 2
                out.append(plate([[x0, yo, z_top], [x1, yo, z_top], [x1 - 0.15, yo, z_top - d], [x0 + 0.1, yo, z_top - d]], 0.03, "gear"))
    return M.Mesh.concat(out)


def details(p, density):
    out = []
    D = p["detail"]
    if "canopy_frame" in D:
        out.extend(canopy_frame(p, density))
    if "probes" in D:
        out.extend(probes(p, density))
    return out
