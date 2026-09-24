# Robot-only rendering (RoboTwin data patch)

KineWorld computes optical flow on a **robot-only** view of each episode — the arm
rendered alone in an empty SAPIEN scene (no objects, table, wall, or ground).
Stock RoboTwin data does not include this view, so these two files re-render it
from the recorded joint trajectories.

They are meant to be dropped into your RoboTwin checkout at the mirrored paths:

| File here | Copy to (in RoboTwin) |
|-----------|-----------------------|
| [`script/render_robot_only.py`](script/render_robot_only.py) | `<RoboTwin>/script/render_robot_only.py` |
| [`envs/utils/robot_only_renderer.py`](envs/utils/robot_only_renderer.py) | `<RoboTwin>/envs/utils/robot_only_renderer.py` |

The second argument is the folder that holds the collected `data/` — i.e. the
KineWorld `<variant>` (e.g. `aloha-agilex_clean_50`), **not** the eval scene config
(`demo_clean`):

```bash
# From your RoboTwin root, after collecting demonstrations for a task:
python script/render_robot_only.py <task> <variant>
# e.g.
python script/render_robot_only.py place_dual_shoes aloha-agilex_clean_50
```

For each episode this writes, next to the original `data/`:

```text
<save_path>/<task>/<variant>/robot_only/
├── data/episode*.hdf5          # robot-only RGB (observation/<camera>/rgb, JPEG-encoded)
└── video/<camera>/episode*.mp4 # per-camera preview
```

`render_robot_only.py` reads `save_path` / `embodiment` from
`./task_config/<variant>.yml` and the robot URDF/camera layout from
`./assets/embodiments/`, using the same camera geometry as the scene renders so
the robot-only frames align with the RGB stream. `robot_only_renderer.py` is the
supporting library (SAPIEN scene + qpos-driven rendering) that the script imports;
it sets qpos directly (no physics) for speed.

The resulting `robot_only/data/*.hdf5` is what training reads when
`FLOW_MODE=robot_only` (see the top-level README's Training section). If the
training data already includes this view, these rendering scripts are not needed.
The renderer follows RoboTwin's `robot_only_gen` interface.
