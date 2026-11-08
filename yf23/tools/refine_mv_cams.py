import sys, json
sys.path.insert(0, "tools")
import multiview as MV
params, align, out = sys.argv[1], sys.argv[2], sys.argv[3]
sel = json.load(open("sfm/nmusaf/mv_selection.json"))
res = MV.refine_cameras("sfm/nmusaf", params, json.load(open(align)), "sfm/nmusaf/masks", set(sel["fit"] + sel["holdout"]))
json.dump(res, open(out, "w"), indent=1)
