"""Compose a labelled progress panel of the overlays for one iteration.

Colours: green = CAD and photo agree, red = photo only (CAD missing),
blue = CAD only (CAD excess), yellow line = CAD outline.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import evaluate as E  # noqa: E402

ORDER = ["side", "top", "top_b", "front"]
TITLES = {"side": "SIDE (fitted)", "top": "TOP (fitted)", "top_b": "TOP hold-out (not fitted)", "front": "FRONT (fitted)"}


def crop_box(name):
    m = cv2.imread(os.path.join(E.REF, E.REFS[name]["mask"]), 0) > 127
    ys, xs = np.nonzero(m)
    pad = 40
    return max(xs.min() - pad, 0), max(ys.min() - pad, 0), xs.max() + pad, ys.max() + pad


def panel(it_dir, out=None, width=900):
    met = json.load(open(os.path.join(it_dir, "metrics.json")))
    tiles = []
    for n in ORDER:
        if n not in met["results"]:
            continue
        im = cv2.imread(os.path.join(it_dir, f"overlay_{n}.jpg"))
        x0, y0, x1, y1 = crop_box(n)
        im = im[y0:y1, x0:x1]
        im = cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA)
        r = met["results"][n]
        bar = np.full((44, width, 3), 255, np.uint8)
        txt = f"{TITLES[n]}   IoU {r['iou']:.4f}   mean edge err {r['boundary_mean_m'] * 100:.1f} cm   p95 {r['boundary_p95_m'] * 100:.0f} cm"
        cv2.putText(bar, txt, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2)
        tiles.append(np.vstack([bar, im]))
    left = np.vstack([t for t, n in zip(tiles, ORDER) if n in ("side", "front")])
    right = np.vstack([t for t, n in zip(tiles, ORDER) if n in ("top", "top_b")])
    h = max(left.shape[0], right.shape[0])
    pad = lambda a: np.vstack([a, np.full((h - a.shape[0], a.shape[1], 3), 255, np.uint8)])
    head = np.full((56, width * 2, 3), 255, np.uint8)
    name = os.path.basename(os.path.normpath(it_dir))
    cv2.putText(head, f"YF-23 CAD  {name}   mean IoU (side, top, front) = {met['mean_iou_fitted']:.4f}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 0, 0), 2)
    leg = "green = agree   red = photo only (CAD missing)   blue = CAD only (excess)"
    cv2.putText(head, leg, (width + 160, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    img = np.vstack([head, np.hstack([pad(left), pad(right)])])
    out = out or os.path.join(it_dir, "panel.jpg")
    cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return out


if __name__ == "__main__":
    print(panel(sys.argv[1]))
