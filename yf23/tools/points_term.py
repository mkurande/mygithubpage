"""Photogrammetry term: robust distance between SfM points (aircraft only) and the CAD surface.

Points come from the CPU COLMAP reconstruction of museum photos of PAV-1, mapped
into the CAD frame by trimmed similarity ICP (sfm/<set>/align_icp.json).
Signed distance is approximated point-to-plane against the nearest face centre
of a dense CAD mesh (good to a few mm at density 0.4).
"""
import json
import numpy as np
import pycolmap
from scipy.spatial import cKDTree

import yf23_model as M
import sfm_align as A


def load_points(workdir, min_track=3, max_err=1.5):
    r = pycolmap.Reconstruction(f"{workdir}/sparse/0")
    pts = [q for q in r.points3D.values() if q.track.length() >= min_track and q.error <= max_err]
    return np.array([q.xyz for q in pts])


def to_cad(P_sfm, al):
    s, R, t = al["s"], np.array(al["R"]), np.array(al["t"])
    return ((P_sfm - t) @ R) / s


def realign(p, P_sfm, al, taus=(0.3, 0.2, 0.15, 0.15)):
    """Re-run trimmed similarity ICP of the CAD (params p) onto the SfM points, starting from al."""
    X = A.surface_samples(p, 150000, gear=True)
    s, R, t = al["s"], np.array(al["R"]), np.array(al["t"])
    for tau in taus:
        for _ in range(3):
            W = s * (X @ R.T) + t
            d, j = cKDTree(W).query(P_sfm)
            sel = d < tau * s
            s, R, t = A.umeyama(X[j[sel]], P_sfm[sel])
    return dict(s=float(s), R=R.tolist(), t=t.tolist()), int(sel.sum())


def select_aircraft(Q, p, band=0.6, zmin=1.0, seed=0, holdout=0.2):
    """Points within `band` m of the current CAD, above the gear zone; split into fit / hold-out."""
    sd = signed_distance(M.build(p, 0.4, gear_down=False), Q)
    keep = (np.abs(sd) < band) & (Q[:, 2] > zmin)
    idx = np.nonzero(keep)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    k = int(len(idx) * holdout)
    return Q[idx[k:]], Q[idx[:k]]


def signed_distance(mesh, Q):
    V = mesh.v[mesh.f]
    c = V.mean(1)
    n = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    _, j = cKDTree(c).query(Q)
    return np.einsum("ij,ij->i", Q - c[j], n[j])


def huber(d, delta=0.10):
    a = np.abs(d)
    return np.where(a < delta, 0.5 * a ** 2 / delta, a - 0.5 * delta)


def point_cost(mesh, Q, cap=0.6):
    d = signed_distance(mesh, Q)
    d = np.clip(d, -cap, cap)
    return float(huber(d).mean()), d
