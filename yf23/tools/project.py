"""Silhouette projection, camera registration and IoU / boundary metrics.

Cameras
-------
A camera is a dict:
  kind: "ortho" | "persp"
  view: "side" | "top" | "front"  (canonical axes, before extra rotation)
  yaw, pitch, roll: degrees, extra rotation of the aircraft about its CG
  f: focal length in pixels (persp only), dist: camera distance in metres (persp)
  scale: pixels per metre (ortho), cx, cy: principal point / image offset (px)
  W, H: image size
Canonical views (image u to the right, v down):
  side  : viewed from port (-y), nose to the LEFT  -> u = x,  v = -z
  top   : viewed from above,     nose UP          -> u = y,  v = x
  front : viewed from ahead,     port on RIGHT    -> u = -y... (pilot's right on image left)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

CG = np.array([10.5, 0.0, 2.0])


def rot(yaw, pitch, roll):
    y, p, r = (math.radians(a) for a in (yaw, pitch, roll))
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    return Rz @ Ry @ Rx


# camera-frame basis per canonical view: rows = (image-right, image-down, depth-away)
BASIS = {
    "side": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float),   # camera at -y looking +y
    "top": np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float),    # camera above looking down, nose up (v = x)
    "front": np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float),  # camera ahead looking aft
}


def project(V, cam):
    R = rot(cam.get("yaw", 0), cam.get("pitch", 0), cam.get("roll", 0))
    P = (V - CG) @ R.T @ BASIS[cam["view"]].T
    if cam["kind"] == "ortho":
        u = cam["cx"] + cam["scale"] * P[:, 0]
        v = cam["cy"] + cam["scale"] * P[:, 1]
        d = P[:, 2]
    else:
        d = P[:, 2] + cam["dist"]
        u = cam["cx"] + cam["f"] * (P[:, 0] + cam.get("tx", 0.0)) / d
        v = cam["cy"] + cam["f"] * cam.get("asp", 1.0) * (P[:, 1] + cam.get("ty", 0.0)) / d
    return np.stack([u, v], -1), d


from numba import njit


@njit(cache=True)
def _raster(tri, H, W):
    """Union-fill of triangles (pixel centres), robust to overlap and winding."""
    m = np.zeros((H, W), np.uint8)
    for t in range(tri.shape[0]):
        x0, y0 = tri[t, 0, 0], tri[t, 0, 1]
        x1, y1 = tri[t, 1, 0], tri[t, 1, 1]
        x2, y2 = tri[t, 2, 0], tri[t, 2, 1]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0.0:
            continue
        s = 1.0 if area > 0 else -1.0
        xmin = max(int(np.floor(min(x0, min(x1, x2)))), 0)
        xmax = min(int(np.ceil(max(x0, max(x1, x2)))), W - 1)
        ymin = max(int(np.floor(min(y0, min(y1, y2)))), 0)
        ymax = min(int(np.ceil(max(y0, max(y1, y2)))), H - 1)
        for y in range(ymin, ymax + 1):
            py = y + 0.5
            for x in range(xmin, xmax + 1):
                px = x + 0.5
                w0 = s * ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py))
                w1 = s * ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py))
                w2 = s * ((x0 - px) * (y1 - py) - (x1 - px) * (y0 - py))
                if w0 >= 0 and w1 >= 0 and w2 >= 0:
                    m[y, x] = 1
    return m


def silhouette(mesh, cam, ss=1):
    """Binary mask (H, W) of the projected mesh. ss = supersampling for AA edges."""
    W, H = cam["W"] * ss, cam["H"] * ss
    c2 = dict(cam)
    for k in ("cx", "cy", "f", "scale"):
        if k in c2:
            c2[k] = c2[k] * ss
    uv, _ = project(mesh.v, c2)
    m = _raster(np.ascontiguousarray(uv[mesh.f], dtype=np.float64), H, W) * 255
    if ss > 1:
        m = cv2.resize(m, (cam["W"], cam["H"]), interpolation=cv2.INTER_AREA)
    return m > 127


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return inter / max(uni, 1)


def contour_pts(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 2))
    return np.vstack([c[:, 0, :] for c in cs]).astype(float)


def boundary_metrics(cad, ref, px_per_m):
    """Symmetric boundary distances in metres: mean, 95th percentile, Hausdorff."""
    dt_ref = cv2.distanceTransform((~edge(ref)).astype(np.uint8), cv2.DIST_L2, 5)
    dt_cad = cv2.distanceTransform((~edge(cad)).astype(np.uint8), cv2.DIST_L2, 5)
    e_cad, e_ref = edge(cad), edge(ref)
    d1 = dt_ref[e_cad]
    d2 = dt_cad[e_ref]
    d = np.concatenate([d1, d2]) / px_per_m
    if d.size == 0:
        return dict(mean=float("nan"), p95=float("nan"), hausdorff=float("nan"))
    return dict(mean=float(d.mean()), p95=float(np.percentile(d, 95)), hausdorff=float(d.max()))


def edge(mask):
    m = mask.astype(np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def overlay(img, cad, ref, alpha=0.55):
    """Green = both, red = reference only (CAD missing), blue = CAD only (excess)."""
    out = img.copy().astype(np.float32) if img is not None else np.full(cad.shape + (3,), 255, np.float32)
    col = np.zeros_like(out)
    both = cad & ref
    col[both] = (60, 200, 60)
    col[ref & ~cad] = (40, 40, 230)
    col[cad & ~ref] = (230, 120, 30)
    any_ = cad | ref
    out[any_] = out[any_] * (1 - alpha) + col[any_] * alpha
    out = out.astype(np.uint8)
    cs, _ = cv2.findContours(cad.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cs, -1, (255, 255, 0), 1)
    return out
