"""Reference silhouette extraction (photo -> binary mask).

Each reference uses a simple, inspectable colour rule followed by morphology,
largest-component selection and hole filling. GrabCut refines the boundary
from that seed. Masks are saved next to the photo and must be visually
checked (refs/masks/*_check.jpg) before they are used for metrics.
"""
import sys

import cv2
import numpy as np


def fill_holes(m):
    m = m.astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)
    return (m | (1 - ff)).astype(bool)


def largest(m):
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return m.astype(bool)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    return lab == k


def grabcut_refine(img, seed, band=12, iters=5):
    fg = cv2.erode(seed.astype(np.uint8), np.ones((band, band), np.uint8))
    bg = ~cv2.dilate(seed.astype(np.uint8), np.ones((band, band), np.uint8)).astype(bool)
    gm = np.full(seed.shape, cv2.GC_PR_BGD, np.uint8)
    gm[seed] = cv2.GC_PR_FGD
    gm[fg > 0] = cv2.GC_FGD
    gm[bg] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(img, gm, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    return (gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)


def rule(img, name):
    b, g, r = [c.astype(int) for c in cv2.split(img)]
    if name.startswith("side"):  # blue sky background
        return ((b - r) < 45) & ((b + g + r) < 450)
    if name.startswith("top"):  # tan desert background, blue-grey aircraft
        return ((b - r) > -4) | ((b + g + r) < 150)
    raise ValueError(name)


HULL_COLS = {"side_pav1_inflight": [(300, 600)]}


def segment(path, name, out):
    img = cv2.imread(path)
    m = rule(img, name)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = fill_holes(largest(m))
    m = fill_holes(largest(grabcut_refine(img, m)))
    for (c0, c1) in HULL_COLS.get(name, []):  # transparent canopy: restore solid bubble
        ys, xs = np.nonzero(m[:, c0:c1])
        hull = cv2.convexHull(np.stack([xs + c0, ys], -1).astype(np.int32))
        hm = np.zeros(m.shape, np.uint8)
        cv2.fillConvexPoly(hm, hull, 1)
        m = m | (hm > 0)
    cv2.imwrite(out, (m * 255).astype(np.uint8))
    chk = img.copy()
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(chk, cs, -1, (0, 255, 255), 2)
    cv2.imwrite(out.replace(".png", "_check.jpg"), chk)
    return m


if __name__ == "__main__":
    for name in sys.argv[1:]:
        m = segment(f"sel/{name}.jpg", name, f"masks/{name}.png")
        print(name, m.sum())
