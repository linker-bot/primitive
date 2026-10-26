# Optional object meshes

Some vision primitives use an object mesh to choose a thumb pre-shape. Store each
model under its target label:

```text
data/mesh_model/{label}/model.obj
```

For an external directory or an installed deployment, set:

```bash
export MESH_MODEL_DIR=/path/to/meshes
```

The mesh loaders also require the optional `trimesh` Python package. Object meshes
are not bundled in the gesture wheel. A missing or unreadable mesh uses the
primitive's default pre-shape; this does not replace its other vision, FK or
feedback requirements. These object models are distinct from the hand URDF/meshes
used for FK and visualization.
