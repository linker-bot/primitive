# Object meshes for vision primitives (optional)

Place per-object models as:

```
data/mesh_model/{label}/model.obj
```

Override root: `export MESH_MODEL_DIR=/path/to/meshes`

Used by force-control `*_by_vision` primitives for thumb_rot pre-shaping when mesh geometry is available. Without mesh, primitives fall back to `object_size` from `bboxes_3d`.
