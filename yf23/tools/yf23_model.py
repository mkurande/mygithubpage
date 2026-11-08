"""Parametric exterior model of the Northrop YF-23A (PAV-1/PAV-2 shared outer mold line).

Coordinate frame (metres): x aft from nose tip, y to starboard, z up.
z = 0 is the ground plane with the gear down (static), so the 4.24 m overall
height is measured to the highest point (V-tail tip).

Every silhouette-defining curve is a control-point table in the params JSON,
interpolated with monotone cubic (PCHIP) splines so refinements stay local.
Components overlap (union of shells); the mesh is for rendering/projection.
The STEP export (export_step.py) builds solids from the same sections.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------
# interpolation helpers
# --------------------------------------------------------------------------
def pchip(xk, yk, x):
    """Monotone piecewise-cubic Hermite interpolation (Fritsch-Carlson)."""
    xk = np.asarray(xk, float)
    yk = np.asarray(yk, float)
    x = np.asarray(x, float)
    n = len(xk)
    if n == 1:
        return np.full_like(x, yk[0])
    h = np.diff(xk)
    d = np.diff(yk) / h
    m = np.zeros(n)
    if n == 2:
        m[:] = d[0]
    else:
        for i in range(1, n - 1):
            if d[i - 1] * d[i] <= 0:
                m[i] = 0.0
            else:
                w1 = 2 * h[i] + h[i - 1]
                w2 = h[i] + 2 * h[i - 1]
                m[i] = (w1 + w2) / (w1 / d[i - 1] + w2 / d[i])
        m[0] = _edge(h[0], h[1], d[0], d[1])
        m[-1] = _edge(h[-1], h[-2], d[-1], d[-2])
    xc = np.clip(x, xk[0], xk[-1])
    idx = np.clip(np.searchsorted(xk, xc) - 1, 0, n - 2)
    t = (xc - xk[idx]) / h[idx]
    h00 = (2 * t**3 - 3 * t**2 + 1)
    h10 = (t**3 - 2 * t**2 + t)
    h01 = (-2 * t**3 + 3 * t**2)
    h11 = (t**3 - t**2)
    return h00 * yk[idx] + h10 * h[idx] * m[idx] + h01 * yk[idx + 1] + h11 * h[idx] * m[idx + 1]


def _edge(h0, h1, d0, d1):
    m = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(m) != np.sign(d0):
        return 0.0
    if np.sign(d0) != np.sign(d1) and abs(m) > abs(3 * d0):
        return 3 * d0
    return m


def table(p, key):
    t = np.asarray(p[key], float)
    return t[:, 0], t[:, 1]


def curve(p, key, x):
    xk, yk = table(p, key)
    return pchip(xk, yk, x)


# --------------------------------------------------------------------------
# mesh container
# --------------------------------------------------------------------------
@dataclass
class Mesh:
    v: np.ndarray  # (N,3) float
    f: np.ndarray  # (M,3) int
    part: np.ndarray  # (M,) int part id

    @staticmethod
    def concat(meshes):
        vs, fs, ps, off = [], [], [], 0
        for m in meshes:
            vs.append(m.v)
            fs.append(m.f + off)
            ps.append(m.part)
            off += len(m.v)
        return Mesh(np.vstack(vs), np.vstack(fs), np.concatenate(ps))


PARTS = ["fuselage", "canopy", "wing", "nacelle", "vtail", "gear", "intake", "frame", "probe"]


def grid_mesh(P, part, close_u=False, cap_start=False, cap_end=False):
    """P: (nv, nu, 3) grid; v runs along the body, u around the section."""
    nv, nu, _ = P.shape
    v = P.reshape(-1, 3)
    ii, jj = np.meshgrid(np.arange(nv - 1), np.arange(nu if close_u else nu - 1), indexing="ij")
    a = ii * nu + jj
    b = ii * nu + (jj + 1) % nu
    c = (ii + 1) * nu + (jj + 1) % nu
    d = (ii + 1) * nu + jj
    f = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    verts = [v]
    faces = [f]
    n0 = len(v)
    for flag, row in ((cap_start, 0), (cap_end, nv - 1)):
        if not flag:
            continue
        ring = np.arange(nu) + row * nu
        ctr = P[row].mean(0)
        verts.append(ctr[None])
        ci = n0
        n0 += 1
        nxt = np.roll(ring, -1)
        faces.append(np.stack([np.full(nu, ci), nxt, ring], -1) if row == 0 else np.stack([np.full(nu, ci), ring, nxt], -1))
    V = np.vstack(verts)
    F = np.vstack(faces)
    return Mesh(V, F, np.full(len(F), PARTS.index(part)))


# --------------------------------------------------------------------------
# section shape: chined superellipse (upper and lower lobes meet at the chine)
# --------------------------------------------------------------------------
def chined_section(w, zc, zt, zb, nu_, nl_, nu, yc=0.0, kchine=0.0, t=None):
    """Closed section; returns (nu, 2) array of (y, z), counter-clockwise from +y chine.

    w: half width at chine, zc: chine height, zt: top, zb: bottom.
    nu_/nl_: superellipse exponents (2=ellipse, larger=boxier, <2=diamond).
    kchine: sharpens the chine by pinching the lobes near the edge (0..1).
    """
    if t is None:
        t = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    up = s >= 0
    ey = np.where(up, 2 / nu_, 2 / nl_)
    y = yc + w * np.sign(c) * np.abs(c) ** ey
    hz = np.where(up, zt - zc, zc - zb)
    zz = np.abs(s) ** ey
    if kchine:
        zz = zz ** (1 + kchine)
    z = zc + np.sign(s) * hz * zz
    return np.stack([y, z], -1)


# --------------------------------------------------------------------------
# adaptive sampling: put stations where the surface bends
# --------------------------------------------------------------------------
def adaptive_stations(p, keys, x0, x1, n, gain=3.0, extra=()):
    """Stations on [x0, x1] with density 1 + gain * normalised |second derivative| of the given curves."""
    xf = np.linspace(x0, x1, 2000)
    act = np.zeros_like(xf)
    for k in keys:
        if k in p:
            y = curve(p, k, xf)
            act += np.abs(np.gradient(np.gradient(y, xf), xf))
    for xe, w in extra:  # point features (e.g. canopy ends, intake lip): local bumps
        act += w * np.exp(-0.5 * ((xf - xe) / 0.25) ** 2)
    act = act / (act.mean() + 1e-12)
    dens = 1.0 + gain * np.clip(act, 0, 8)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(xf))])
    cdf /= cdf[-1]
    return np.interp(np.linspace(0, 1, n), cdf, xf)


def chine_clustered(nu, c=0.55):
    """Section angles clustered towards the chines (t = 0 and pi)."""
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    return u - 0.5 * c * np.sin(2 * u)


# --------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------
def fuselage(p, nx, nu):
    x0, x1 = p["fus_x"]
    # cosine clustering at the nose for curvature resolution
    s = np.linspace(0, 1, nx)
    D = p.get("detail")
    if D:
        xs = adaptive_stations(p, ["fus_halfwidth", "fus_top", "fus_bottom", "fus_chine_z"], x0, x1, nx,
                               extra=[(p["can_x"][0], 2.0), (p["can_x"][1], 1.0), (0.0, 4.0)])
        tt = chine_clustered(nu)
    else:
        xs = x0 + (x1 - x0) * (0.5 - 0.5 * np.cos(np.pi * s))
        tt = None
    w = np.maximum(curve(p, "fus_halfwidth", xs), 1e-3)
    zt = curve(p, "fus_top", xs)
    zb = curve(p, "fus_bottom", xs)
    zc = curve(p, "fus_chine_z", xs)
    zt = np.maximum(zt, zc + 1e-3)
    zb = np.minimum(zb, zc - 1e-3)
    eu = curve(p, "fus_exp_upper", xs)
    el = curve(p, "fus_exp_lower", xs)
    P = np.zeros((nx, nu, 3))
    for i, x in enumerate(xs):
        sec = chined_section(w[i], zc[i], zt[i], zb[i], eu[i], el[i], nu, kchine=p.get("chine_sharp", 0.0), t=tt)
        P[i, :, 0] = x
        P[i, :, 1:] = sec
    return grid_mesh(P, "fuselage", close_u=True, cap_end=True)


def fus_surface_z(p, x, y):
    """Height of the fuselage upper surface at station x and lateral position y (|y| <= half width)."""
    x = np.atleast_1d(np.asarray(x, float))
    w = np.maximum(curve(p, "fus_halfwidth", x), 1e-3)
    zt, zc = curve(p, "fus_top", x), curve(p, "fus_chine_z", x)
    eu = curve(p, "fus_exp_upper", x)
    k = p.get("chine_sharp", 0.0)
    c = np.clip(np.abs(np.asarray(y, float)) / w, 0, 1) ** (eu / 2)
    sn = np.sqrt(np.clip(1 - c ** 2, 0, 1))
    return zc + (np.maximum(zt, zc) - zc) * (sn ** (2 / eu)) ** (1 + k)


def canopy(p, nx, nu):
    x0, x1 = p["can_x"]
    s = np.linspace(0, 1, nx)
    xs = x0 + (x1 - x0) * (0.5 - 0.5 * np.cos(np.pi * s))
    w = np.maximum(curve(p, "can_halfwidth", xs), 1e-3)
    zt = curve(p, "can_top", xs)
    zb = curve(p, "fus_top", xs) - 0.05
    zt = np.maximum(zt, zb + 1e-3)
    P = np.zeros((nx, nu, 3))
    t = np.linspace(0, np.pi, nu)
    seat = p.get("detail") is not None  # detail models seat the canopy on the real fuselage surface
    for i, x in enumerate(xs):
        c, sn = np.cos(t), np.sin(t)
        e = 2 / p["can_exp"]
        yy = w[i] * np.sign(c) * np.abs(c) ** e
        P[i, :, 0] = x
        P[i, :, 1] = yy
        if seat:
            base = fus_surface_z(p, np.full_like(yy, x), yy) - 0.02
            P[i, :, 2] = base + np.maximum(zt[i] - base[nu // 2], 0) * np.abs(sn) ** e
        else:
            P[i, :, 2] = zb[i] + (zt[i] - zb[i]) * np.abs(sn) ** e
    return grid_mesh(P, "canopy")


def airfoil_thickness(xc, tc):
    """Biconvex-ish half-thickness distribution (fraction of chord)."""
    return tc * 2.0 * xc * (1 - xc) * (1 + 0.15 * (1 - 2 * xc))


def wing(p, ns, nc):
    """Diamond wing. Both halves; panel from y_root to the tip."""
    L = p["wing"]
    tl, tt = math.tan(math.radians(L["le_sweep_deg"])), math.tan(math.radians(L["te_sweep_deg"]))
    y0, yt = L["y_root"], L["semispan"]
    meshes = []
    ys = np.linspace(y0, yt, ns)
    xc = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nc))
    for sgn in (1, -1):
        P = np.zeros((ns, 2 * nc - 2, 3))
        for i, y in enumerate(ys):
            xle = L["x_le_c"] + y * tl
            xte = L["x_te_c"] - y * tt
            ch = xte - xle
            frac = (y - y0) / (yt - y0)
            tc = L["tc_root"] + (L["tc_tip"] - L["tc_root"]) * frac
            zc = L["z_root"] + (y - y0) * math.tan(math.radians(L["dihedral_deg"])) - L["twist_lift"] * frac
            ht = airfoil_thickness(xc, tc) * ch
            xu = xle + xc * ch
            up = np.stack([xu, np.full(nc, sgn * y), zc + ht], -1)
            lo = np.stack([xu[::-1], np.full(nc, sgn * y), zc - ht[::-1]], -1)
            P[i] = np.vstack([up[:-1], lo[:-1]])
        meshes.append(grid_mesh(P, "wing", close_u=True, cap_end=True))
    return Mesh.concat(meshes)


def nacelles(p, nx, nu):
    N = p["nacelle"]
    D = p.get("detail") or {}
    meshes = []
    x0, x1 = N["x"]
    s = np.linspace(0, 1, nx)
    xs = x0 + (x1 - x0) * s
    if D:
        xs = adaptive_stations(p, ["nac_y", "nac_halfwidth", "nac_top", "nac_bottom"], x0, x1, nx,
                               extra=[(x0, 6.0), (D.get("trough", {}).get("x0", x1), 2.0)])
    yc = curve(p, "nac_y", xs)
    w = np.maximum(curve(p, "nac_halfwidth", xs), 1e-3)
    zt = curve(p, "nac_top", xs)
    zb = curve(p, "nac_bottom", xs)
    el = curve(p, "nac_exp", xs)
    et = curve(p, "nac_exp_top", xs) if "nac_exp_top" in p else el
    legacy = "nac_bottom_ratio" not in p  # iteration-00 parameter files
    br = curve(p, "nac_bottom_ratio", xs) if not legacy else np.ones_like(xs)
    sh = curve(p, "nac_bottom_shift", xs) if not legacy else np.zeros_like(xs)
    tr = D.get("trough")
    for sgn in (1, -1):
        P = np.zeros((nx, nu, 3))
        for i, x in enumerate(xs):
            zc = 0.5 * (zt[i] + zb[i])
            sec = chined_section(w[i], zc, zt[i], zb[i], et[i], el[i], nu)
            yy, zz = sec[:, 0], sec[:, 1]
            h = (zz - zb[i]) / max(zt[i] - zb[i], 1e-3)  # 0 at bottom, 1 at top
            if legacy:
                yy = np.where(yy > 0, yy * (1 - N["outboard_cant"] * h), yy * (1 - N["inboard_cant"] * h))
            else:
                scale = br[i] + (1 - br[i]) * h            # bottom narrower when br < 1
                yy = yy * scale + sh[i] * (1 - h)          # sh: inboard(-)/outboard(+) shift of the bottom
            if tr and x > tr["x0"]:
                # recessed exhaust trough in the upper deck, ramping down towards the nacelle end
                ramp = min(1.0, (x - tr["x0"]) / max(tr["ramp"], 1e-3))
                u = np.abs(yy) / (tr["half_width_frac"] * w[i])
                cup = np.clip(1 - u ** 8, 0, 1) * (zz > zc)
                zz = zz - tr["depth"] * ramp * cup
            P[i, :, 0] = x
            P[i, :, 1] = sgn * (yc[i] + yy)
            P[i, :, 2] = zz
        if sgn < 0:
            P = P[:, ::-1]
        intake = D.get("intake")
        meshes.append(grid_mesh(P, "nacelle", close_u=True, cap_start=not intake, cap_end=True))
        if intake:
            meshes.extend(intake_duct(P, xs, intake))
    return Mesh.concat(meshes)


def intake_duct(P, xs, I):
    """Open intake: lip ring, inner duct running aft, engine face with a hub."""
    k = max(3, int(np.searchsorted(xs, xs[0] + I["duct_len"])))
    ring0 = P[0]
    ctr = ring0.mean(0)
    rad = np.linalg.norm(ring0[:, 1:] - ctr[1:], axis=1).mean()
    f = max(0.2, 1 - I["lip"] / rad)
    inner = np.empty((k, P.shape[1], 3))
    for i in range(k):
        c = P[i].mean(0)
        shrink = f * (1 - I.get("taper", 0.15) * i / max(k - 1, 1))
        inner[i] = c + (P[i] - c) * shrink
        inner[i, :, 0] = P[i, :, 0] + I["lip"] * 0.5
    lip = np.stack([ring0, inner[0]], 0)
    out = [grid_mesh(lip, "nacelle", close_u=True),
           grid_mesh(inner[:, ::-1], "intake", close_u=True)]  # reversed winding: faces inward
    face = inner[-1]
    c = face.mean(0)
    hub = c + np.array([-0.25, 0, 0])
    out.append(grid_mesh(np.stack([face[::-1], c + (face[::-1] - c) * 0.3, np.repeat(hub[None], len(face), 0)], 0), "intake", close_u=True))
    return out


def vtail_chord(T, r):
    """Leading/trailing edge (x) at span station r for the pentagon V-tail."""
    tl = math.tan(math.radians(T["le_sweep_deg"]))
    if "te_sweep_deg" in T:  # iteration-00 trapezoid tail
        return T["x_le_root"] + r * tl, T["x_le_root"] + T["root_chord"] + r * math.tan(math.radians(T["te_sweep_deg"]))
    R = T["span"]
    xle = T["x_le_root"] + r * tl
    x_root_te = T["x_le_root"] + T["root_chord"]
    x_tip_le = T["x_le_root"] + R * tl
    x_tip_te = x_tip_le + T["tip_chord"]
    rk = T["kink_frac"] * R
    x_k = T["x_le_root"] + T["kink_x"]
    xte = np.interp(r, [0.0, rk, R], [x_root_te, x_k, x_tip_te])
    return xle, xte


def vtails(p, ns, nc):
    T = p["vtail"]
    cant = math.radians(T["cant_from_vertical_deg"])
    span = T["span"]
    xc = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, nc))
    meshes = []
    rs = np.unique(np.concatenate([np.linspace(0, span, ns), [T.get("kink_frac", 0.5) * span]]))
    for sgn in (1, -1):
        P = np.zeros((len(rs), 2 * nc - 2, 3))
        for i, r in enumerate(rs):
            xle, xte = vtail_chord(T, r)
            ch = xte - xle
            ht = airfoil_thickness(xc, T["tc"]) * ch
            xu = xle + xc * ch
            ry, rz = math.sin(cant), math.cos(cant)
            ny, nz = math.cos(cant), -math.sin(cant)
            by, bz = T["y_root"] + r * ry, T["z_root"] + r * rz
            up = np.stack([xu, by + ht * ny, bz + ht * nz], -1)
            lo = np.stack([xu[::-1], by - ht[::-1] * ny, bz - ht[::-1] * nz], -1)
            sec = np.vstack([up[:-1], lo[:-1]])
            sec[:, 1] *= sgn
            P[i] = sec if sgn > 0 else sec[::-1]
        meshes.append(grid_mesh(P, "vtail", close_u=True, cap_start=True, cap_end=True))
    return Mesh.concat(meshes)


def exhaust_sidewalls(p):
    """Trough sidewalls that protrude aft of the nacelle end, forming the W-shaped aft deck."""
    E = p["exhaust"]
    meshes = []
    for wall in E["walls"]:
        for sgn in (1, -1):
            y, x0, x1, xt = wall["y"], wall["x0"], wall["x1"], wall["x_tip"]
            zt, zb, t = wall["z_top"], wall["z_bot"], E["thickness"]
            # planform: quad from (x0..x1) at full height, tapering to a tip at xt (top edge)
            P = np.array([
                [[x0, y - t, zb], [x0, y + t, zb], [x0, y + t, zt], [x0, y - t, zt]],
                [[x1, y - t, zb], [x1, y + t, zb], [x1, y + t, zt], [x1, y - t, zt]],
                [[xt, y - t * 0.3 + wall.get("tip_dy", 0.0), zt - 0.05], [xt, y + t * 0.3 + wall.get("tip_dy", 0.0), zt - 0.05],
                 [xt, y + t * 0.3 + wall.get("tip_dy", 0.0), zt], [xt, y - t * 0.3 + wall.get("tip_dy", 0.0), zt]],
            ], float)
            P[..., 1] *= sgn
            if sgn < 0:
                P = P[:, ::-1]
            meshes.append(grid_mesh(P, "nacelle", close_u=True, cap_start=True, cap_end=True))
    return Mesh.concat(meshes)


def cylinder(p0, p1, r, nu=24, nv=4):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    a = p1 - p0
    L = np.linalg.norm(a)
    a /= L
    tmp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, tmp)
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    t = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    P = np.zeros((nv, nu, 3))
    for i, s in enumerate(np.linspace(0, 1, nv)):
        P[i] = p0 + a * L * s + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * w)
    return P


def gear(p, res=24):
    G = p["gear"]
    meshes = []
    for g in G["legs"]:
        for sgn in ((1, -1) if g["y"] > 0 else (1,)):
            y = sgn * g["y"]
            # strut
            P = cylinder([g["x"], y, g["z_top"]], [g["x"], y, g["wheel_r"]], g["strut_r"], res, 3)
            meshes.append(grid_mesh(P, "gear", close_u=True))
            # wheel (axis along y)
            P = cylinder([g["x"], y - g["wheel_w"] / 2, g["wheel_r"]], [g["x"], y + g["wheel_w"] / 2, g["wheel_r"]], g["wheel_r"], res * 2, 3)
            meshes.append(grid_mesh(P, "gear", close_u=True, cap_start=True, cap_end=True))
            # door (thin plate, fore-aft)
            if g.get("door_h", 0) > 0:
                d = g["door_h"]
                x0, x1 = g["x"] - g["door_len"] / 2, g["x"] + g["door_len"] / 2
                yo = y + sgn * g.get("door_dy", 0.25)
                Pd = np.array([[[x0, yo, g["z_top"]], [x1, yo, g["z_top"]]],
                               [[x0, yo, g["z_top"] - d], [x1, yo, g["z_top"] - d]]])
                meshes.append(grid_mesh(Pd, "gear"))
    return Mesh.concat(meshes)


def build(p, density=1.0, gear_down=True):
    """density scales every sampling count; 1.0 ≈ 0.3 M triangles."""
    k = lambda n: max(4, int(round(n * density)))
    parts = [
        fuselage(p, k(400), k(160)),
        canopy(p, k(120), k(60)),
        wing(p, k(120), k(80)),
        nacelles(p, k(260), k(96)),
        vtails(p, k(80), k(60)),
    ]
    if "exhaust" in p:
        parts.append(exhaust_sidewalls(p))
    if p.get("detail"):
        import yf23_detail as YD
        parts.extend(YD.details(p, density))
        if gear_down:
            parts.append(YD.gear_detailed(p, density))
    elif gear_down:
        parts.append(gear(p, k(16)))
    return Mesh.concat(parts)


def load_params(path):
    with open(path) as fh:
        return json.load(fh)
