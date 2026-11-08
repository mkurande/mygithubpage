"""SAM (ViT-B, CPU) segmentation from box / point prompts. Embeddings cached per image."""
import os, sys, json
import numpy as np, cv2, torch
from segment_anything import sam_model_registry, SamPredictor

_P = None
def predictor():
    global _P
    if _P is None:
        sam = sam_model_registry["vit_b"](checkpoint=os.path.join(os.path.dirname(__file__), "..", "models", "sam_vit_b_01ec64.pth"))
        sam.eval(); torch.set_num_threads(4)
        _P = SamPredictor(sam)
    return _P

def segment_all(img_bgr, box=None, pos=(), neg=()):
    """All three SAM candidates (masks, scores)."""
    P = predictor()
    P.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pts = list(pos) + list(neg)
    lab = [1] * len(pos) + [0] * len(neg)
    m, s, _ = P.predict(point_coords=np.array(pts, float) if pts else None, point_labels=np.array(lab) if pts else None,
                        box=np.array(box, float) if box is not None else None, multimask_output=True)
    return m, s


def segment(img_bgr, box=None, pos=(), neg=(), multimask=True):
    P = predictor()
    P.set_image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    pts = list(pos) + list(neg)
    lab = [1] * len(pos) + [0] * len(neg)
    m, s, _ = P.predict(point_coords=np.array(pts, float) if pts else None, point_labels=np.array(lab) if pts else None,
                        box=np.array(box, float) if box is not None else None, multimask_output=multimask)
    i = int(np.argmax(s))
    return m[i], float(s[i])
