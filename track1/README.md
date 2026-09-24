# KineWorld · WorldArena2 Track 1

此目录实现动作条件视频推理：读取每条 HDF5 轨迹的 14 维动作和目标帧数，逐段生成 9 个关键帧，按 `visual_stride=4` 扩展到原始时间轴，并生成规定帧数的 H.264 视频。默认 `action_flow` 模式要求逐段提供真实动作光流；`zero_flow` 只作为显式选择的文本条件基线。

## 预计算动作光流

需要能够运行 SAPIEN/Vulkan 渲染和 RAFT 的环境。示例：

```bash
python track1/precompute_action_flow.py \
  --dataset-root /path/to/dataset_track1 \
  --output-root /path/to/action_flow \
  --robotwin-assets-root /path/to/RoboTwin \
  --render-width 640 --render-height 480 \
  --target-width 320 --target-height 240 \
  --episode-start 1 --episode-end 1000 \
  --flow-device cuda:0
```

预计算结果保留 320×240 光流图像及字节哈希；推理时在内存中调整到 Wan VAE 对齐所需的 320×256，不改写源 PNG。

## 生成视频

本源码包不带训练权重。KineWorld 推理必须显式指定本地权重，或者同时指定远端仓库及文件名。

```bash
python track1/infer_track1.py \
  --dataset-root /path/to/dataset_track1 \
  --output-root /path/to/kineworld_run \
  --checkpoint-path /path/to/checkpoint.safetensors \
  --conditioning-mode action_flow \
  --action-flow-provider precomputed \
  --precomputed-flow-root /path/to/action_flow \
  --device cuda:0
```

可以用 `--dry-run` 检查 episode 发现与分片，且不会加载模型。

## 输出校验

```bash
python track1/validate_track1.py outputs \
  --dataset-root /path/to/dataset_track1 \
  --videos-dir /path/to/kineworld_run/videos \
  --episode-start 1 --episode-end 1000 \
  --output-json /path/to/kineworld_run/validation_report.json
```

校验包括解码、帧数、分辨率、帧率、首帧和动作条件记录。模型质量需要另行使用对应评测器测量。
