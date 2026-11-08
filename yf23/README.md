# YF-23 exterior CAD: projection fit

A parametric exterior model of the Northrop YF-23A, fitted so that its side, top
and front silhouettes match real photographs. Built from scratch in this repo;
nothing is carried over from the earlier ChatGPT-hosted project, which this
environment could not reach.

**Best result: `iter_07`, mean silhouette IoU 0.962 (side 0.971, top 0.974,
front 0.943); never-fitted top hold-out photo 0.976. Baseline `iter_00`: 0.864.**
Progress page: published as a private Claude artifact (see the PR description).

| Iter | Mean | Side | Top | Front | Top hold-out | Change |
|---|---|---|---|---|---|---|
| 00 | 0.8635 | 0.847 | 0.882 | 0.862 | 0.882 | published dimensions only |
| 01 | 0.9072 | 0.901 | 0.938 | 0.883 | 0.938 | structure fixes + back-projected photo measurements |
| 02 | 0.9465 | 0.961 | 0.970 | 0.909 | 0.971 | first optimizer pass (114 control points) |
| 03 | 0.9535 | 0.961 | 0.970 | 0.929 | 0.973 | optimizer |
| 04 | 0.9557 | 0.960 | 0.971 | 0.936 | 0.974 | optimizer |
| 05 | 0.9599 | 0.969 | 0.972 | 0.939 | 0.975 | denser control points (193), curve-shift moves |
| 06 | 0.9603 | 0.967 | 0.973 | 0.941 | 0.975 | optimizer |
| **07** | **0.9623** | **0.971** | **0.974** | **0.943** | **0.976** | higher-resolution objective |
| 08 | 0.9621 | 0.971 | 0.974 | 0.942 | 0.976 | no gain |
| 09 | 0.9621 | 0.970 | 0.973 | 0.943 | 0.976 | no gain (plateau) |

All rows use one registration protocol (`YF23_WORK_SCALE=0.5`). Full numbers,
including edge distances in metres, are in `iterations/history.json` and each
`iterations/iter_NN/metrics.json`.

## Layout

- `tools/yf23_model.py`: parametric generator (chined fuselage, canopy, diamond wing,
  trapezoid-intake nacelles, pentagon V-tails, exhaust trough sidewalls, gear).
- `cad/params/iter_NN.json`: one parameter file per revision (the CAD source).
- `tools/project.py`: cameras, numba silhouette rasteriser, IoU and boundary metrics.
- `tools/evaluate.py`: camera registration and scoring against each reference.
- `tools/refine.py`: constrained pattern search on shape parameters (cameras fixed).
- `tools/run_iteration.py`: refine, re-register, evaluate, panel, history.
- `refs/`: selected photos (`sel/`), masks (`masks/`), the hand trace for the front view
  (`front_trace.json`), Commons metadata (`meta/`).
- `iterations/iter_NN/`: metrics, cameras, overlay panel; full-size overlays kept for 00 and 07.
- `exports/yf23_iter07_preview.glb`: 0.36 M-triangle mesh, gear down.
- `report/`: the progress page source.

## Reproduce

```
pip install numpy scipy opencv-python-headless numba trimesh
cd tools
python fetch_refs.py ../refs/raw ../refs/commons_titles.txt   # optional; selected photos are committed
YF23_WORK_SCALE=0.5 python evaluate.py ../cad/params/iter_07.json ../iterations/iter_07 --init ../iterations/iter_07/cameras.json
YF23_WORK_SCALE=0.5 python run_iteration.py 10              # next iteration from iter_09
python export_mesh.py ../cad/params/iter_07.json out.glb 3.0   # ~3.2 M triangles
```

## Method

1. Register a perspective camera to each photo (pose within ±12°, position,
   focal length; pixel aspect within ±5% for scanned prints).
2. With cameras fixed, pattern-search the shape control points to raise the mean
   IoU of side, top and front, with a smoothness penalty. Nose position, overall
   length (20.55 m) and semispan (6.64 m) are locked, so photos constrain
   proportions and published figures set size.
3. Re-register the cameras and score at full resolution.

The top hold-out photo is never used for fitting; it checks for overfitting.

## Limitations

- IoU measures silhouettes only, not surfaces inside the outline.
- Camera pose and shape are fitted together, so some error can sit in the camera.
- The fitted top photo is only 640 × 480 (served upscaled). The 1800 px top photo
  is the hold-out. Swap their roles in the next round.
- The front reference is a close-range, cluttered museum photo, hand-traced (trace
  revision v2 in `refs/front_trace.json`). Gear is excluded.
- No photogrammetry, STEP export, mobile feedback app or submission trigger yet.

## Reference credits

| View | Photo | Credit | Licence |
|---|---|---|---|
| Side | [YF-23 left view](https://commons.wikimedia.org/wiki/File:YF-23_left_view.jpg) | U.S. Air Force photo (060810-F-1234S-007) | Public domain |
| Top (fitted) | [Yf-23 1](https://commons.wikimedia.org/wiki/File:Yf-23_1.jpg) | U.S. Navy | Public domain |
| Top (hold-out) | [YF-23 top view](https://commons.wikimedia.org/wiki/File:YF-23_top_view.jpg) | U.S. Air Force photo (060810-F-1234S-004) | Public domain |
| Front | [NMUSAF head-on](https://commons.wikimedia.org/wiki/File:Northrop_McDonnell_Douglas_YF-23A_PAV-1_87-0800_Black_Widow_II_HeadOn_R%26D_NMUSAF_25Sep09_(14599837582).jpg) | Valder137 | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |

Overlay images derived from the front photo are adaptations under CC BY 2.0.
Dimensions: length 67 ft 5 in (20.55 m), span 43 ft 7 in (13.28 m), height
13 ft 11 in (4.24 m), from [FAS](https://man.fas.org/dod-101/sys/ac/f-23.htm)
and [GlobalSecurity](https://www.globalsecurity.org/military/systems/aircraft/f-23-specs.htm).
