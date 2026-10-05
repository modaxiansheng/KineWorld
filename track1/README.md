# KineWorld · WorldArena2 Track 1

此目录实现动作条件视频推理：读取每条 HDF5 轨迹的 14 维动作和目标帧数，逐段生成 9 个关键帧，按 `visual_stride=4` 扩展到原始时间轴，并生成规定帧数的 H.264 视频。默认 `action_flow` 模式要求逐段提供真实动作光流；`zero_flow` 只作为显式选择的文本条件基线。

## 预计算动作光流

需要能够运行 SAPIEN/Vulkan 渲染和 RAFT 的环境。示例：

```bash
python track1/precompute_action_flow.py \
  --dataset-root /path/to/dataset_track1 \
  --output-root /path/to/action_flow \
  --robotwin-assets-root /path/to/RoboTwin/assets \
  --render-width 640 --render-height 480 \
  --target-width 320 --target-height 240 \
  --episode-start 1 --episode-end 1000 \
  --flow-device cuda:0
```

预计算结果保留 320×240 光流图像及字节哈希；推理时在内存中调整到 Wan VAE 对齐所需的 320×256，不改写源 PNG。

## 生成视频

使用公开的 [KineWorld step-500 权重](https://huggingface.co/pumpkin601/KineWorld)。请先按照主 README 的 [Model download](../README.md#model-download) 下载权重和 Wan 基础模型。完整的输入格式、单条试运行和常见问题见 [checkpoint 使用说明](../docs/checkpoint_usage.md)。

从仓库根目录运行。先只生成 episode1，确认输出后再扩展到 1--1000；当前适配器在选择 episode 前仍会检查完整的 1,000 条输入文件布局。

```bash
python track1/infer_track1.py \
  --dataset-root /path/to/dataset_track1 \
  --output-root outputs/kineworld-step500-episode1 \
  --checkpoint-path checkpoints/KineWorld/step-500.safetensors \
  --model-cache-dir models \
  --conditioning-mode action_flow \
  --action-flow-provider precomputed \
  --precomputed-flow-root /path/to/action_flow \
  --episode-start 1 --episode-end 1 \
  --num-inference-steps 25 --seed 1 \
  --device cuda:0
```

也可以把 `--checkpoint-path ...` 替换为 `--checkpoint-repo pumpkin601/KineWorld --checkpoint-file step-500.safetensors --checkpoint-revision 37e8f86c6c3cf45cde743162bf9b6de583cf1b73`，由脚本自动下载。两种权重来源不能同时指定。视频推理不需要 `action_norm_stats.npz`，该文件仅用于动作策略入口。

可以用 `--dry-run` 检查 episode 发现与分片，且不会加载模型或校验动作光流文件；真实推理时不要加此参数。

## 输出校验

```bash
python track1/validate_track1.py outputs \
  --dataset-root /path/to/dataset_track1 \
  --videos-dir outputs/kineworld-step500-episode1/videos \
  --records-dir outputs/kineworld-step500-episode1/per_episode \
  --run-config outputs/kineworld-step500-episode1/run_config.json \
  --episode-start 1 --episode-end 1 \
  --output-json outputs/kineworld-step500-episode1/validation_report.json
```

校验包括解码、帧数、分辨率、帧率、首帧和动作条件记录。模型质量需要另行使用对应评测器测量。
