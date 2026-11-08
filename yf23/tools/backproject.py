"""Back-project photo pixels through a registered camera onto an aircraft-frame plane."""
import numpy as np
import project as P


def backproject(cam, u, v, normal, offset):
    R = P.rot(cam.get("yaw", 0), cam.get("pitch", 0), cam.get("roll", 0))
    B = P.BASIS[cam["view"]]
    M = B @ R  # aircraft (relative to CG) -> camera
    C = np.array([-cam.get("tx", 0.0), -cam.get("ty", 0.0), -cam["dist"]])
    d = np.array([(u - cam["cx"]) / cam["f"], (v - cam["cy"]) / cam["f"], 1.0])
    # back to aircraft frame
    Ca = M.T @ C + P.CG
    da = M.T @ d
    n = np.asarray(normal, float)
    t = (offset - n @ Ca) / (n @ da)
    return Ca + t * da
