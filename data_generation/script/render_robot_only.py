"""
Re-render every collected episode in a robot-only minimal SAPIEN scene
(no objects, table, wall, ground) and write the results to
``<save_path>/<task_name>/<task_config>/robot_only/``.

Output layout (matches the official RoboTwin robot_only format):
    robot_only/
        data/episode{i}.hdf5          # observation/{cam}/rgb (JPEG-encoded)
        video/{cam}/episode{i}.mp4    # per-camera mp4

Usage:
    python script/render_robot_only.py <task_name> <task_config>
"""

import os
import sys
import warnings

sys.path.append("./")

from argparse import ArgumentParser

import h5py
import numpy as np
import yaml

from envs.utils.images_to_video import images_to_video
from envs.utils.pkl2hdf5 import images_encoding
from envs.utils.robot_only_renderer import (
    DEFAULT_CAMERA_CONFIG,
    ROBOT_NAME_MAP,
    RobotOnlyScene,
    load_embodiment_config,
)


def _episode_index(filename: str) -> int:
    return int(filename.replace("episode", "").replace(".hdf5", ""))


def render_one_episode(
    scene: RobotOnlyScene,
    in_hdf5: str,
    out_hdf5: str,
    out_video_root: str,
) -> int:
    """Render one episode and write robot_only hdf5 + per-camera mp4."""
    with h5py.File(in_hdf5, "r") as f:
        left_arm = f["joint_action/left_arm"][:]
        right_arm = f["joint_action/right_arm"][:]
        left_gripper = f["joint_action/left_gripper"][:]
        right_gripper = f["joint_action/right_gripper"][:]

    T = left_arm.shape[0]
    camera_names = scene.camera_names_ordered
    frames_per_cam = {name: [] for name in camera_names}

    for t in range(T):
        rgb_dict = scene.set_pose_and_render(
            left_arm[t], right_arm[t], left_gripper[t], right_gripper[t]
        )
        for name in camera_names:
            frames_per_cam[name].append(rgb_dict[name])

    os.makedirs(os.path.dirname(out_hdf5), exist_ok=True)
    with h5py.File(out_hdf5, "w") as f:
        obs = f.create_group("observation")
        for name in camera_names:
            cam_group = obs.create_group(name)
            encode_data, max_len = images_encoding(frames_per_cam[name])
            cam_group.create_dataset("rgb", data=encode_data, dtype=f"S{max_len}")

    ep_basename = os.path.basename(out_hdf5).replace(".hdf5", ".mp4")
    for name in camera_names:
        video_path = os.path.join(out_video_root, name, ep_basename)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            images_to_video(np.array(frames_per_cam[name]), out_path=video_path)

    return T


def main(task_name: str, task_config: str) -> None:
    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    save_path = cfg["save_path"]
    embodiment = cfg["embodiment"]
    if len(embodiment) not in (1, 3):
        raise ValueError(
            "robot_only rendering supports embodiment configs of length 1 or 3, "
            f"got {embodiment}"
        )
    # For dual_arm shared embodiment (len==1) and asymmetric (len==3), use the
    # primary (left) robot as the rendered embodiment.
    robot_prefix = embodiment[0]
    if robot_prefix not in ROBOT_NAME_MAP:
        raise ValueError(
            f"Unsupported embodiment '{robot_prefix}'. "
            f"Known: {list(ROBOT_NAME_MAP.keys())}"
        )

    robotwin_assets = os.path.abspath("./assets")
    embodiment_config, robot_dir = load_embodiment_config(robotwin_assets, robot_prefix)

    camera_config_dict = {k: v.copy() for k, v in DEFAULT_CAMERA_CONFIG.items()}

    task_root = os.path.join(save_path, task_name, task_config)
    in_data_dir = os.path.join(task_root, "data")
    out_root = os.path.join(task_root, "robot_only")
    out_data_dir = os.path.join(out_root, "data")
    out_video_root = os.path.join(out_root, "video")

    if not os.path.isdir(in_data_dir):
        print(f"[render_robot_only] No data dir at {in_data_dir}, nothing to do.")
        return

    episode_files = sorted(
        [
            f
            for f in os.listdir(in_data_dir)
            if f.startswith("episode") and f.endswith(".hdf5")
        ],
        key=_episode_index,
    )
    if not episode_files:
        print(f"[render_robot_only] No episode hdf5 found under {in_data_dir}.")
        return

    print(
        f"[render_robot_only] task={task_name} config={task_config} "
        f"episodes={len(episode_files)} embodiment={robot_prefix}"
    )

    scene = RobotOnlyScene(embodiment_config, robot_dir, camera_config_dict)
    scene.setup()

    try:
        for ep_file in episode_files:
            ep_idx = _episode_index(ep_file)
            in_hdf5 = os.path.join(in_data_dir, ep_file)
            out_hdf5 = os.path.join(out_data_dir, ep_file)
            ep_basename_mp4 = ep_file.replace(".hdf5", ".mp4")

            all_videos_exist = all(
                os.path.exists(os.path.join(out_video_root, name, ep_basename_mp4))
                for name in scene.camera_names_ordered
            )
            if os.path.exists(out_hdf5) and all_videos_exist:
                print(f"[render_robot_only] skip episode {ep_idx} (already done)")
                continue

            T = render_one_episode(scene, in_hdf5, out_hdf5, out_video_root)
            print(f"[render_robot_only] episode {ep_idx} rendered ({T} frames)")
    finally:
        scene.close()

    print("[render_robot_only] Done.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parsed = parser.parse_args()
    main(parsed.task_name, parsed.task_config)
