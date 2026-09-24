"""
RoboTwin evaluation integration for the KineWorld flow-action policy.

Implements the four functions RoboTwin's ``script/eval_policy.py`` expects:
  - encode_obs(observation)  -> (img_arr, state)
  - get_model(usr_args)      -> model
  - eval(TASK_ENV, model, observation)
  - reset_model(model)

Runs in the RoboTwin conda environment. Model inference happens on a remote
server over WebSocket (see ../flow_action_server.py).
"""

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from kineworld_model import KineWorldPolicy


def encode_obs(observation):
    """Extract camera images and joint state from a RoboTwin observation.

    Returns:
        img_arr: [head_rgb, right_rgb, left_rgb]  — each (H, W, 3) uint8
        state:   (14,) float32 joint vector
    """
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]
    input_state = observation["joint_action"]["vector"]
    return input_rgb_arr, input_state


def get_model(usr_args):
    """Instantiate the remote KineWorld policy client.

    Reads server connection info from usr_args (set via deploy_policy.yml and
    eval.sh --overrides).
    """
    host = usr_args.get("server_host", "0.0.0.0")
    port = int(usr_args.get("server_port", 8000))
    chunk = int(usr_args.get("action_chunk_size", 49))
    overlap = int(usr_args.get("overlap", 0))
    task_name = usr_args.get("task_name")

    model = KineWorldPolicy(
        host=host,
        port=port,
        action_chunk_size=chunk,
        overlap=overlap,
        task_name=task_name,
    )
    # When True (default), skip the ray-traced get_obs() between consecutive
    # actions inside one replan window; the model only consumes the latest
    # frame, which is refreshed at the end of the chunk. Set false in
    # deploy_policy.yml for a full-FPS eval video.
    model.skip_get_obs_within_replan = bool(usr_args.get(
        "skip_get_obs_within_replan", True
    ))
    return model


def eval(TASK_ENV, model, observation):
    """Run one inference step: predict an action chunk, execute all actions.

    The server returns absolute qpos actions whose first entry (action[0]) is
    the GT-anchored current qpos. That anchor is skipped here; only the future
    actions are executed.
    """
    if model.instruction is None:
        instruction = TASK_ENV.get_instruction()
        model.set_language(instruction)

    # Cold-start: seed the cache with the current observation only if none is
    # cached yet. On later chunks the cache already holds the observation
    # pushed at the end of the previous chunk's execution loop.
    if not model.has_history():
        input_rgb_arr, input_state = encode_obs(observation)
        model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()
    pixel_prefix = min(getattr(model, "pixel_prefix", 0), len(actions))
    actions_to_execute = actions[pixel_prefix:]
    print(
        f"[KineWorld] Action chunk: total={len(actions)}, "
        f"prefix_skip={pixel_prefix}, execute={len(actions_to_execute)}"
    )

    # Refresh the observation only after the LAST action of the chunk (which
    # feeds the next chunk's inference); the intermediate renders are
    # redundant because the model only consumes the latest frame. Set
    # skip_get_obs_within_replan=false to render every step.
    skip = bool(getattr(model, "skip_get_obs_within_replan", True))
    n = len(actions_to_execute)
    for i, action in enumerate(actions_to_execute):
        TASK_ENV.take_action(action)
        is_last = (i == n - 1)
        if TASK_ENV.eval_success:
            break
        if (not skip) or is_last:
            observation = TASK_ENV.get_obs()
            input_rgb_arr, input_state = encode_obs(observation)
            model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model):
    """Clear all episode state (server buffers + local caches)."""
    model.reset()
