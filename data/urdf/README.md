# Hand URDF models (required for FK / RViz)

Place the official LinkerHand URDF (with meshes) here:

```
data/urdf/{model}/{side}/linkerhand_{model}_{side}.urdf
```

Versioned filenames are also accepted when resolving a model and side,
e.g. `linkerhand_l6v3.1_left.urdf` satisfies a request for `l6` / `left`.

Override root: `export LINKERHAND_URDF_DIR=/path/to/linkerhand-urdf`

For reproducible deployments, use the bundled model or explicitly configure
`LINKERHAND_URDF_DIR`. Legacy workspace search paths are only a compatibility fallback.

An existing file passed through the node's `urdf_path` parameter takes precedence
over discovery. A missing path does not select a model; verify it with `test -f`.

## Bundled models in installed deployments

The repository includes the O6 **right** URDF and its meshes. Wheels and colcon
installations preserve this tree under `share/hand_gesture_primitives/urdf`;
source distributions carry the same files under their project-local `data/urdf`.
There is no second resource tree in the source checkout. The isolated Jazzy
installer copies and hashes these assets along with the Python and message packages.

Discovery supports the repository layout, `LINKERHAND_URDF_DIR`, the ament package
share directory, `sys.prefix/share/hand_gesture_primitives/urdf`, and legacy workspace
layouts. Use the node's explicit `urdf_path` when an exact model path must be selected.

Left O6 and the other models must be supplied separately; the right model is not
mirrored or substituted for another hand. Configure external models with:

```bash
export LINKERHAND_URDF_DIR=/path/to/linkerhand-urdf
```

Pinocchio remains an optional FK dependency. In the isolated Jazzy environment,
use the ROS/system package rather than replacing apt NumPy/SciPy with pip builds.

## Without a URDF

A missing FK geometry model affects the features below. Radian command conversion
can still require a separate controller `robot_description`:

- `gesture_node` — fingertip topic/markers are unavailable; FK-dependent primitives require an appropriate model
- `joint_state_bridge` — exits with an error (its only job is driving the mesh model via FK)
