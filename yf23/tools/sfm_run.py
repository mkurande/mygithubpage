"""CPU structure-from-motion with pycolmap: SIFT features, exhaustive matching, incremental mapping.
usage: python sfm_run.py <workdir>   (expects <workdir>/images)"""
import sys, os, time
import pycolmap

wd = sys.argv[1]
img, db, out = os.path.join(wd, "images"), os.path.join(wd, "database.db"), os.path.join(wd, "sparse")
os.makedirs(out, exist_ok=True)
t = time.time()
if not os.path.exists(db):
    eo = pycolmap.FeatureExtractionOptions()
    eo.sift.max_num_features = 8192
    eo.max_image_size = 1600
    pycolmap.extract_features(db, img, extraction_options=eo, camera_mode=pycolmap.CameraMode.PER_IMAGE)
    print(f"features done {time.time()-t:.0f}s", flush=True)
    pycolmap.match_exhaustive(db)
    print(f"matching done {time.time()-t:.0f}s", flush=True)
maps = pycolmap.incremental_mapping(db, img, out)
print(f"mapping done {time.time()-t:.0f}s", flush=True)
for k, r in maps.items():
    print(k, r.summary(), flush=True)
