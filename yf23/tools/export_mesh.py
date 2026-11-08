"""Export a parameter set as GLB/STL. usage: python export_mesh.py params.json out.glb [density] [--no-gear]
density 1.0 ~ 0.36 M triangles, 3.0 ~ 3.2 M, 9.0 ~ 29 M (sampling only; fidelity is set by the parameters)."""
import sys
import trimesh
import yf23_model as M

if __name__ == "__main__":
    p = M.load_params(sys.argv[1])
    d = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 1.0
    m = M.build(p, d, gear_down="--no-gear" not in sys.argv)
    trimesh.Trimesh(m.v, m.f, process=False).export(sys.argv[2])
    print(len(m.f), "triangles ->", sys.argv[2])
