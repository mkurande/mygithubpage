"""Align the CAD into the SfM frame: initial pose from a shared camera, then trimmed similarity ICP."""
import json, sys
import numpy as np
import pycolmap
from scipy.spatial import cKDTree
sys.path.insert(0, "tools")
import project as P
import yf23_model as M


def cad_to_cam(cam):
    """Return (Mx, m) with camera coords Pc = Mx @ X + m (metres) for our perspective camera."""
    R = P.rot(cam.get("yaw", 0), cam.get("pitch", 0), cam.get("roll", 0))
    Mx = P.BASIS[cam["view"]] @ R
    m = -Mx @ P.CG + np.array([cam.get("tx", 0.0), cam.get("ty", 0.0), cam["dist"]])
    return Mx, m


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    Dm = np.eye(3); Dm[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ Dm @ Vt
    s = np.trace(np.diag(sig) @ Dm) / (S ** 2).sum(1).mean()
    t = mu_d - s * R @ mu_s
    return s, R, t


def surface_samples(p, n=200000, gear=True, seed=0):
    m = M.build(p, 1.0, gear_down=gear)
    V = m.v[m.f]
    a = 0.5 * np.linalg.norm(np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), n, p=a / a.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    T = V[idx]
    return T[:, 0] + u[:, None] * (T[:, 1] - T[:, 0]) + v[:, None] * (T[:, 2] - T[:, 0])
