"""
Pre-encode T5 text embeddings, VAE latents, and RAFT flow for RoboTwin training.

Runs in two phases per rank:

  Phase 1 (text, fast):
      Loads T5 + WanPrompter, batch-encodes all 100 instructions per episode
      into dual streams (video_contexts = CAMERA_PREFIX + instr, action_contexts
      = bare instr), trims trailing zeros, and writes the result into
      {cache_root}/{task}/{variant}/{episode}.pt via atomic tmp+rename.

  Phase 2 (vision, slow):
      Unloads T5, loads VAE + RAFT, iterates all assigned episodes, runs the
      existing VAE+flow pipeline, and merges the rgb/flow latents + metadata
      into the same .pt (preserving the Phase 1 text fields).

Each phase has its own tqdm bar on rank 0 and is independently idempotent
(re-running skips episodes that already have the relevant fields), so a
killed job can resume cleanly.

Supports multi-node multi-GPU via torchrun (see precompute.sh).
Sharding is purely env-driven through RANK / WORLD_SIZE — the script never
calls torch.distributed.init_process_group.
"""

import argparse
import gc
import json
import glob
import os
import sys
import time
from typing import Dict, List, Optional

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from io import BytesIO
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from reversible_flow_codec import FlowCodec
from raft_flow_extractor import RAFTFlowExtractor
from flow_prefix_utils import (
    tile_flow_t_shape,
    tile_flow_with_white_wrist,
    resize_flow,
    threshold_flow_magnitude,
    encode_flows_to_pil,
    compute_flows,
    add_bg_texture,
    mask_flows_by_robot,
    process_camera_flow,
    process_camera_flow_full_scene,
)
from dataset_action_robotwin import (
    tshape_tile,
    ROBOTWIN_ALL_TASKS,
    RoboTwinActionFlowDataset,
)
from diffsynth.models.model_manager import ModelManager
from diffsynth.prompters.wan_prompter import WanPrompter

WRIST_MAG_RATIO = 1.5
# Cache-version tags stamped into each .pt file. Bump when the text (Phase 1)
# or vision (Phase 2) encoding behaviour changes in a way that makes older
# cache files stale; .pt files lacking the current tag are re-encoded
# atomically (the other phase's fields are preserved through the merge).
TEXT_EMBEDDING_VERSION = 1
PHASE2_VERSION = 4


def parse_args():
    p = argparse.ArgumentParser(description="Pre-encode VAE/text latents for RoboTwin")
    p.add_argument("--dataset_base_path", type=str, required=True)
    p.add_argument("--cache_root", type=str, required=True,
                    help="Output directory for cached .pt files")
    p.add_argument("--vae_path", type=str, required=True,
                    help="Path to Wan2.2_VAE.pth")
    p.add_argument("--text_encoder_path", type=str, required=True,
                    help="Path to models_t5_umt5-xxl-enc-bf16.pth")
    p.add_argument("--tokenizer_dir", type=str, required=True,
                    help="Path to google/umt5-xxl tokenizer dir")
    p.add_argument("--variants", nargs="+",
                    default=["aloha-agilex_clean_50", "aloha-agilex_randomized_500"])
    p.add_argument("--task_names", nargs="*", default=None,
                    help="Tasks to process (empty = all 50)")
    p.add_argument("--cameras", nargs="+",
                    default=["head_camera", "left_camera", "right_camera"])
    p.add_argument("--num_frames", type=int, default=49,
                    help="Action sequence length (metadata only; actions are "
                         "loaded live from HDF5 at train time, not cached).")
    # --- Visual temporal downsampling (MUST match the training dataset) ---
    # dataset_action_robotwin.__getitem__ samples video/flow frames as
    # ``start + i*visual_stride`` for i in range(num_video_frames), while the
    # action chunk stays high-FPS (``start + i`` for num_frames). Defaults
    # (visual_stride=1, num_video_frames=None -> num_frames) reproduce the
    # legacy 1:1 behaviour. For the current downsample run use
    # visual_stride=4, num_video_frames=13.
    p.add_argument("--num_video_frames", type=int, default=None,
                    help="Number of stride-visual_stride video/flow frames per "
                         "chunk. None -> num_frames (legacy 1:1).")
    p.add_argument("--visual_stride", type=int, default=1,
                    help="Frame stride for video/flow sampling. MUST match the "
                         "training dataset's visual_stride.")
    p.add_argument("--size_w", type=int, default=320)
    p.add_argument("--size_h", type=int, default=256)
    p.add_argument("--flow_method", type=str, default="raft")
    p.add_argument("--flow_mode", type=str, default="robot_only",
                    choices=["robot_only", "full_scene"])
    p.add_argument("--flow_max_magnitude", type=float, default=15.0)
    p.add_argument("--flow_noise_threshold", type=float, default=0.5)
    p.add_argument("--chunk_stride", type=int, default=1,
                    help="Stride for precomputed VAE chunks. stride=1 stores every "
                         "possible start index (max diversity, max disk/time); "
                         "stride=k stores every k-th start index (k× faster, k× "
                         "smaller). Adjacent stride-1 chunks overlap ~98%% so "
                         "stride=2 is usually a free win.")
    p.add_argument("--skip_phase1", action="store_true",
                    help="Skip T5 text encoding (e.g. resuming VAE-only rerun)")
    p.add_argument("--skip_phase2", action="store_true",
                    help="Skip VAE+RAFT encoding (e.g. text-only rerun)")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────
#  Episode discovery
# ────────────────────────────────────────────────────────────────

def discover_all_episodes(base_path, variants, task_names, flow_mode):
    """Return list of dicts with episode metadata, across all variants."""
    if not task_names:
        task_names = ROBOTWIN_ALL_TASKS
    episodes = []
    for variant in variants:
        require_ro = (flow_mode == "robot_only")
        for task in sorted(task_names):
            variant_dir = os.path.join(base_path, task, variant)
            data_dir = os.path.join(variant_dir, "data")
            ro_dir = os.path.join(variant_dir, "robot_only", "data")
            if not os.path.isdir(data_dir):
                continue
            if require_ro and not os.path.isdir(ro_dir):
                continue
            for hdf5_path in sorted(glob.glob(os.path.join(data_dir, "episode*.hdf5"))):
                ep_name = os.path.splitext(os.path.basename(hdf5_path))[0]
                ro_path = os.path.join(ro_dir, f"{ep_name}.hdf5")
                episodes.append({
                    "task": task,
                    "variant": variant,
                    "episode_name": ep_name,
                    "data_hdf5": hdf5_path,
                    "robot_only_hdf5": ro_path if os.path.exists(ro_path) else None,
                    "variant_dir": variant_dir,
                })
    return episodes


# ────────────────────────────────────────────────────────────────
#  Persistence helpers
# ────────────────────────────────────────────────────────────────

def _pt_path_for(cache_root: str, ep: Dict) -> str:
    out_dir = os.path.join(cache_root, ep["task"], ep["variant"])
    return os.path.join(out_dir, f"{ep['episode_name']}.pt")


def _save_atomic(path: str, obj: Dict):
    """Write torch.save to a tmp file then os.replace -> atomic on same FS."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _load_existing(path: str) -> Dict:
    """Return existing .pt content or {} if the file is missing."""
    if not os.path.exists(path):
        return {}
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[Cache] WARN failed to read {path}: {e}. Treating as empty.")
        return {}


def _phase1_done(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        "video_contexts" in d
        and "action_contexts" in d
        and d.get("text_embedding_version") == TEXT_EMBEDDING_VERSION
    )


def _phase2_done(path: str, expected_stride: int = 1,
                 expected_visual_stride: int = 1,
                 expected_num_video_frames: Optional[int] = None) -> bool:
    """Return True iff `path` already has stride/visual/version-matching data.

    Older .pt files (pre ``--chunk_stride``) implicitly used stride=1, so a
    missing ``chunk_stride`` field is treated as 1 for backward compatibility.
    Similarly ``phase2_version`` defaults to 1 for legacy files. A mismatch in
    any field forces an atomic re-encode that preserves the Phase 1 text
    fields already stored alongside.

    IMPORTANT: ``visual_stride``/``num_video_frames`` are checked so a 1:1
    cache is never mistaken for a downsample cache (or vice-versa) at the same
    ``chunk_stride``. Legacy files without these fields default to the 1:1
    convention (visual_stride=1).
    """
    if not os.path.exists(path):
        return False
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if "rgb_input_latents" not in d or "flow_input_latents" not in d:
        return False
    stored_stride = int(d.get("chunk_stride", 1))
    if stored_stride != int(expected_stride):
        return False
    stored_visual_stride = int(d.get("visual_stride", 1))
    if stored_visual_stride != int(expected_visual_stride):
        return False
    if expected_num_video_frames is not None:
        stored_nvf = int(d.get("num_video_frames", 0))
        if stored_nvf and stored_nvf != int(expected_num_video_frames):
            return False
    stored_version = int(d.get("phase2_version", 1))
    return stored_version == PHASE2_VERSION


# ────────────────────────────────────────────────────────────────
#  Text encoding helpers (Phase 1)
# ────────────────────────────────────────────────────────────────

def _trim_list(emb_batched: torch.Tensor) -> List[torch.Tensor]:
    """Detect per-item seq_len via nonzero-row mask (encode_prompt zeros
    positions past seq_len), return list of (seq_len_i, D) bf16 CPU tensors.
    """
    assert emb_batched.dim() == 3, f"expected (K, L, D) got {emb_batched.shape}"
    mask = (emb_batched.abs().sum(dim=-1) > 0)  # (K, L) bool
    seq_lens = mask.sum(dim=1).tolist()
    return [
        emb_batched[i, :L].detach().to("cpu", dtype=torch.bfloat16).clone()
        for i, L in enumerate(seq_lens)
    ]


def _trim_list_from_mask(
    emb_batched: torch.Tensor, seq_lens: torch.Tensor,
) -> List[torch.Tensor]:
    """Same output as ``_trim_list`` but uses the attention mask directly
    instead of scanning for non-zero rows. Used by the fast-path encoder
    where we know each prompt's true token count up front.
    """
    assert emb_batched.dim() == 3, f"expected (K, L, D) got {emb_batched.shape}"
    seq_lens_list = seq_lens.tolist()
    return [
        emb_batched[i, :int(L)].detach().to("cpu", dtype=torch.bfloat16).clone()
        for i, L in enumerate(seq_lens_list)
    ]


def _encode_prompts_fast(
    prompter, prompts: List[str], device,
) -> (torch.Tensor, torch.Tensor):
    """Drop-in replacement for ``WanPrompter.encode_prompt(prompts, ...)``
    that uses dynamic padding instead of padding to ``text_len=512``.

    Why this matters: ``WanPrompter`` defaults to ``padding='max_length'``
    with ``max_length=512`` so the tokenizer pads every sequence to 512
    tokens regardless of actual content. T5 self-attention is O(N^2) in
    sequence length, so for RoboTwin prompts (CAMERA_PREFIX + instruction
    ≈ 80 tokens) this does ~25x more work than necessary.

    This path instead pads each batch to ``max(actual_len)`` only, caps
    hard at 512 to match the existing schema, and returns *both* the full
    ``(K, L_dyn, D)`` embeddings *and* the per-item ``seq_lens`` so the
    caller can trim without re-scanning for non-zero rows. Bit-identity
    with the stock path is preserved at the content positions because T5
    attention is already mask-aware.
    """
    prompts = prompter.process_prompt(prompts, positive=True)
    # The underlying HF tokenizer's __call__ accepts kwargs that override
    # the WanPrompter defaults; passing padding='longest' switches it to
    # dynamic padding. We still cap at text_len (512) for safety.
    ids, mask = prompter.tokenizer(
        prompts, return_mask=True, add_special_tokens=True,
        padding="longest", truncation=True,
        max_length=prompter.text_len,
    )
    ids = ids.to(device)
    mask = mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = prompter.text_encoder(ids, mask)
    # Zero positions past each item's true seq_len. Note: the reference
    # ``WanPrompter.encode_prompt`` has a per-batch off-by-one bug here
    # (uses ``prompt_emb[:, v:] = 0`` which zeros across the whole batch
    # for every iteration, effectively clamping to min(seq_lens)). We fix
    # it here with per-item indexing so every prompt keeps its true
    # content; ``_trim_list_from_mask`` downstream still only keeps the
    # in-mask positions so the on-disk cache is unaffected by that
    # upstream bug regardless.
    for i, v in enumerate(seq_lens):
        prompt_emb[i, int(v):] = 0
    return prompt_emb, seq_lens


def _load_instructions(ep: Dict) -> List[str]:
    """Return the 100 'seen' instructions; fall back to task name on missing."""
    instr_file = os.path.join(
        ep["variant_dir"], "instructions", f"{ep['episode_name']}.json",
    )
    fallback = [ep["task"].replace("_", " ")]
    if not os.path.exists(instr_file):
        return fallback
    try:
        with open(instr_file, "r") as f:
            jdata = json.load(f)
    except Exception as e:
        print(f"[Phase1] WARN failed to parse {instr_file}: {e}")
        return fallback
    seen = jdata.get("seen") or []
    return seen if len(seen) > 0 else fallback


def run_phase1_text(
    args, my_episodes: List[Dict], dev, dt,
    rank: int, local_rank: int, node_rank: int,
):
    """Phase 1: encode all instructions per episode, save atomically.

    ``local_rank`` drives per-machine tqdm visibility so every node shows a
    progress bar (not just global rank 0, which would only be visible on the
    master machine's terminal).
    """
    camera_prefix = RoboTwinActionFlowDataset.CAMERA_PREFIX

    # Filter pending episodes (resume support)
    pending = [
        ep for ep in my_episodes
        if not _phase1_done(_pt_path_for(args.cache_root, ep))
    ]
    total = len(my_episodes)
    skipped = total - len(pending)
    if local_rank == 0:
        print(f"[Phase1][node{node_rank}] {skipped}/{total} episodes already "
              f"have text on rank {rank}; {len(pending)} pending")

    if not pending:
        if local_rank == 0:
            print(f"[Phase1][node{node_rank}] Nothing to do.")
        return

    print(f"[Phase1] Rank {rank}: Loading T5 from {args.text_encoder_path} ...")
    mm_text = ModelManager()
    mm_text.load_model(args.text_encoder_path, device=dev, torch_dtype=dt)
    text_encoder = mm_text.fetch_model("wan_video_text_encoder")
    assert text_encoder is not None, "Failed to load T5 text encoder"

    prompter = WanPrompter(tokenizer_path=args.tokenizer_dir, text_len=512)
    prompter.fetch_models(text_encoder)
    print(f"[Phase1] Rank {rank}: T5 + prompter ready")

    # Show one progress bar per machine (local_rank==0). The bar aggregates
    # only the local rank's own ``pending`` list, so the ETA on each node
    # reflects what that rank still has to do. The desc includes both node
    # and global rank so mixed logs are still decipherable.
    pbar = tqdm(
        total=len(pending),
        desc=f"[node{node_rank} R{rank}] Phase1 text",
        unit="ep", disable=(local_rank != 0), dynamic_ncols=True,
        position=0, leave=True,
    )
    t0 = time.time()
    try:
        for ep in pending:
            pt_path = _pt_path_for(args.cache_root, ep)
            pbar.set_postfix_str(f"{ep['task']}/{ep['variant']}/{ep['episode_name']}")

            instructions = _load_instructions(ep)
            video_prompts = [camera_prefix + s for s in instructions]
            action_prompts = list(instructions)

            try:
                with torch.no_grad():
                    # Dynamic-padding fast path: T5 self-attention is O(N^2)
                    # and WanPrompter's default padding='max_length' (512)
                    # wastes ~25x compute on short robot instructions. See
                    # _encode_prompts_fast for details.
                    vctx_b, vlens = _encode_prompts_fast(
                        prompter, video_prompts, device=dev,
                    )
                    actx_b, alens = _encode_prompts_fast(
                        prompter, action_prompts, device=dev,
                    )
            except Exception as e:
                print(f"[Phase1] Rank {rank}: FAILED encode "
                      f"{ep['task']}/{ep['variant']}/{ep['episode_name']}: {e}")
                pbar.update(1)
                continue

            video_contexts = _trim_list_from_mask(vctx_b, vlens)
            action_contexts = _trim_list_from_mask(actx_b, alens)

            existing = _load_existing(pt_path)
            existing.update({
                "video_contexts":          video_contexts,
                "action_contexts":         action_contexts,
                "camera_prefix":           camera_prefix,
                "text_embedding_version":  TEXT_EMBEDDING_VERSION,
                "instructions":            instructions,
                "task":                    ep["task"],
                "variant":                 ep["variant"],
                "episode_name":            ep["episode_name"],
            })
            _save_atomic(pt_path, existing)
            pbar.update(1)
    finally:
        pbar.close()

    elapsed = time.time() - t0
    if local_rank == 0:
        print(f"[Phase1][node{node_rank}] Rank {rank}: {len(pending)} "
              f"episodes in {elapsed:.0f}s "
              f"({elapsed/max(len(pending),1):.2f}s/ep)")

    # Free T5 before Phase 2 to reclaim ~11 GB VRAM
    del text_encoder, prompter, mm_text
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ────────────────────────────────────────────────────────────────
#  Frame helpers (used by Phase 2)
# ────────────────────────────────────────────────────────────────

def load_rgb_frame(f, camera, idx, size):
    jpeg = f[f"observation/{camera}/rgb"][idx]
    img = Image.open(BytesIO(bytes(jpeg)))
    # The hdf5 jpeg was written with cv2.imencode() on RGB frames (OpenCV treats
    # the input as BGR), so PIL decodes it back as BGR. Swap R/B here to recover
    # TRUE RGB, matching the pretrained WAN colour space. MUST stay in sync with
    # dataset_action_robotwin._load_hdf5_rgb_frame and the inference client.
    return np.ascontiguousarray(
        np.array(img.resize(size, Image.BICUBIC), dtype=np.uint8)[..., ::-1])


def load_rgb_frame_raw(f, camera, idx):
    jpeg = f[f"observation/{camera}/rgb"][idx]
    img = Image.open(BytesIO(bytes(jpeg)))
    # See load_rgb_frame: undo the cv2.imencode BGR/RGB swap -> true RGB.
    return np.ascontiguousarray(
        np.array(img.convert("RGB"), dtype=np.uint8)[..., ::-1])


# ────────────────────────────────────────────────────────────────
#  VAE helpers
# ────────────────────────────────────────────────────────────────

def preprocess_image_bf16(pil_img, w, h):
    """PIL -> (1, C, H, W) bf16 in [-1,1]."""
    arr = np.array(pil_img.resize((w, h), Image.BICUBIC), dtype=np.float32)
    t = torch.from_numpy(arr).to(dtype=torch.bfloat16).permute(2, 0, 1).unsqueeze(0)
    t = t * (2.0 / 255.0) - 1.0
    return t


def preprocess_video_bf16(pil_list, w, h):
    """List[PIL] -> (C, T, H, W) bf16 in [-1,1]."""
    frames = [preprocess_image_bf16(p, w, h) for p in pil_list]
    vid = torch.cat(frames, dim=0)   # (T, C, H, W)
    vid = vid.squeeze(1)
    vid = vid.permute(1, 0, 2, 3)   # (C, T, H, W)
    return vid


def encode_chunk(vae, video_pil_list, dev, dt, target_w, target_h):
    """Encode a chunk of PIL images -> latent with first-frame swap."""
    vid = preprocess_video_bf16(video_pil_list, target_w, target_h)
    first = preprocess_image_bf16(video_pil_list[0], target_w, target_h)
    first = first.transpose(0, 1)  # (C, 1, H, W)

    with torch.no_grad():
        z = vae.encode([vid], device=dev).to(dtype=dt, device=dev)
        fz = vae.encode([first], device=dev).to(dtype=dt, device=dev)
    z[:, :, 0:1] = fz
    return z.squeeze(0).cpu()


# ────────────────────────────────────────────────────────────────
#  Per-episode VAE+flow processing (Phase 2)
# ────────────────────────────────────────────────────────────────

def process_episode_vision(
    ep, args, vae, raft_ext, codec, dev, dt, target_w, target_h,
):
    """Encode all chunks' RGB+flow latents for one episode.

    DOWNSAMPLE-ALIGNED with dataset_action_robotwin.__getitem__: each chunk's
    video/flow spans ``num_video_frames`` frames at ``visual_stride`` spacing,
    and flow is computed PER-CHUNK via the shared ``process_camera_flow`` over
    exactly those frames (so add_bg_texture / robot-mask background detection
    is per-chunk, bit-identical to online). This makes the cached latents
    drop-in compatible with the current online run (safe to resume from).
    """

    visual_stride = max(1, int(getattr(args, "visual_stride", 1)))
    _nvf = getattr(args, "num_video_frames", None)
    num_video_frames = int(_nvf) if _nvf else int(args.num_frames)

    f_data = h5py.File(ep["data_hdf5"], "r")
    f_ro = None
    if args.flow_mode == "robot_only" and ep["robot_only_hdf5"]:
        f_ro = h5py.File(ep["robot_only_hdf5"], "r")

    try:
        head_cam = args.cameras[0]
        T = f_data[f"observation/{head_cam}/rgb"].shape[0]
        # Tail-pad: every frame except the very last (T-1) is a valid chunk
        # start. Chunks whose 49-frame window runs past the episode end are
        # padded by clamping `frame_indices` to T-1 below — same trick the
        # action loader has always used. This keeps the trailing motion in
        # the cache instead of dropping it on the floor.
        n_chunks = max(1, T - 1)
        size = (args.size_w, args.size_h)
        head_size = size
        half_size = (args.size_w // 2, args.size_h // 2)

        # ---- Load ALL RGB frames for the episode ----
        all_head = [load_rgb_frame(f_data, args.cameras[0], t, size) for t in range(T)]
        all_left = [load_rgb_frame(f_data, args.cameras[1], t, size) for t in range(T)]
        all_right = [load_rgb_frame(f_data, args.cameras[2], t, size) for t in range(T)]
        all_tiled_rgb = [
            Image.fromarray(tshape_tile(h, l, r))
            for h, l, r in zip(all_head, all_left, all_right)
        ]

        # ---- Pre-load native frames for PER-CHUNK flow ----
        # DOWNSAMPLE ALIGNMENT: flow is NOT computed once over the whole
        # episode's consecutive frames and then sliced. It is computed
        # per-chunk over the num_video_frames stride-visual_stride frames in
        # the loop below, using the SAME process_camera_flow[_full_scene] as
        # the online dataset. add_bg_texture / robot-mask detect background
        # from each CHUNK's first frame, so an episode-level flow can't be
        # reused. We only pre-load the native frames here (cheap, in RAM).
        if args.flow_mode == "full_scene":
            fs_head_native = [load_rgb_frame_raw(f_data, args.cameras[0], t) for t in range(T)]
            fs_left_native = [load_rgb_frame_raw(f_data, args.cameras[1], t) for t in range(T)]
            fs_right_native = [load_rgb_frame_raw(f_data, args.cameras[2], t) for t in range(T)]
        else:
            assert f_ro is not None
            ro_native_all = [load_rgb_frame_raw(f_ro, head_cam, t) for t in range(T)]

    finally:
        f_data.close()
        if f_ro is not None:
            f_ro.close()

    # ---- Precompute episode-level flow pairs (robot_only fast path) ----
    # OPTIMIZATION, bit-identical to the per-chunk path for robot_only: the
    # pure-color background makes detect_bg_color constant across frames, so
    # every chunk's flow[j] = flow(a -> b) with b = a + visual_stride (or a
    # clamped duplicate a==b) can be computed ONCE per UNIQUE (a,b) via the SAME
    # shared funcs (add_bg_texture -> compute_flows -> mask_flows_by_robot ->
    # resize_flow -> encode_flows_to_pil) and reused. Across all chunks these
    # pairs repeat ~num_video_frames x visual_stride times, so this cuts the
    # RAFT + CPU-postproc calls by ~10x. VAE encode stays PER-CHUNK (causal).
    pair_flow_pil = {}
    white_sentinel_pil = None
    if args.flow_mode != "full_scene":
        textured_all = add_bg_texture(ro_native_all)  # bg from frame[0], per-frame
        _wh, _hh = head_size
        white_sentinel_pil = Image.fromarray(
            np.full((_hh, _wh, 3), 255, dtype=np.uint8)
        )
        _cs = max(1, int(getattr(args, "chunk_stride", 1)))
        _pairs = set()
        for _start in range(0, n_chunks, _cs):
            _vid = [min(T - 1, _start + i * visual_stride)
                    for i in range(num_video_frames)]
            for _k in range(num_video_frames - 1):
                _pairs.add((_vid[_k], _vid[_k + 1]))
        for (_a, _b) in _pairs:
            _flow = compute_flows(
                [textured_all[_a], textured_all[_b]],
                flow_method=args.flow_method, raft_extractor=raft_ext,
            )[0]
            _flow = mask_flows_by_robot([_flow], [ro_native_all[_a]])[0]
            _flow = resize_flow(_flow, head_size)
            _pil = encode_flows_to_pil(
                codec, [_flow], max_magnitude=args.flow_max_magnitude,
            )[0][0]
            pair_flow_pil[(_a, _b)] = _pil

    # ---- Encode chunks through VAE at the requested stride ----
    #
    # Start indices span [0, n_chunks) where n_chunks = max(1, T - 1) (see
    # above). Chunks where start + num_frames > T have their frame indices
    # clamped to T-1, i.e. the trailing slots are padded by repeating the
    # last real frame. The action loader applies the same clamp so RGB,
    # flow, and target qpos stay aligned.
    #
    # stride=1 stores every possible start index (max coverage, max cost).
    # stride>1 stores every k-th start index; adjacent stride-1 chunks share
    # 48/49 frames so downsampling rarely affects training quality while
    # linearly shrinking both compute and disk.
    chunk_rgb_latents = []
    chunk_flow_latents = []
    chunk_starts = []

    stride = max(1, int(getattr(args, "chunk_stride", 1)))
    for start in range(0, n_chunks, stride):
        # Video/flow frame indices: num_video_frames stride-visual_stride
        # frames (matches dataset_action_robotwin.__getitem__), tail-clamped.
        vid_indices = [
            min(T - 1, start + i * visual_stride)
            for i in range(num_video_frames)
        ]

        chunk_rgb_pil = [all_tiled_rgb[fi] for fi in vid_indices]

        # PER-CHUNK flow over exactly those stride-visual_stride frames — the
        # identical code path to online _process_camera_flow. The white
        # sentinel at index 0 is prepended INSIDE process_camera_flow[_full_scene].
        if args.flow_mode == "full_scene":
            head_flow, _ = process_camera_flow_full_scene(
                [fs_head_native[fi] for fi in vid_indices],
                target_size=head_size, codec=codec,
                flow_method=args.flow_method, raft_extractor=raft_ext,
                max_magnitude=args.flow_max_magnitude,
                noise_threshold=args.flow_noise_threshold,
            )
            left_flow, _ = process_camera_flow_full_scene(
                [fs_left_native[fi] for fi in vid_indices],
                target_size=half_size, codec=codec,
                flow_method=args.flow_method, raft_extractor=raft_ext,
                max_magnitude=args.flow_max_magnitude * WRIST_MAG_RATIO,
                noise_threshold=args.flow_noise_threshold,
            )
            right_flow, _ = process_camera_flow_full_scene(
                [fs_right_native[fi] for fi in vid_indices],
                target_size=half_size, codec=codec,
                flow_method=args.flow_method, raft_extractor=raft_ext,
                max_magnitude=args.flow_max_magnitude * WRIST_MAG_RATIO,
                noise_threshold=args.flow_noise_threshold,
            )
            chunk_flow_pil = tile_flow_t_shape(
                head_flow, left_flow, right_flow, target_size=head_size,
            )
        else:
            # Reuse episode-level pair flows (bit-identical to per-chunk
            # process_camera_flow for robot_only; white sentinel at index 0,
            # then flow(vid[k] -> vid[k+1]) for each consecutive video frame).
            head_flow_pil = [white_sentinel_pil]
            for _k in range(num_video_frames - 1):
                head_flow_pil.append(
                    pair_flow_pil[(vid_indices[_k], vid_indices[_k + 1])]
                )
            chunk_flow_pil = tile_flow_with_white_wrist(
                head_flow_pil, target_size=head_size,
            )

        rgb_lat = encode_chunk(vae, chunk_rgb_pil, dev, dt, target_w, target_h)
        flow_lat = encode_chunk(vae, chunk_flow_pil, dev, dt, target_w, target_h)

        chunk_rgb_latents.append(rgb_lat)
        chunk_flow_latents.append(flow_lat)
        chunk_starts.append(start)

    return {
        "rgb_input_latents":  chunk_rgb_latents,
        "flow_input_latents": chunk_flow_latents,
        "start_indices":      chunk_starts,
        "episode_length":     T,
        "num_chunks":         len(chunk_starts),
        "chunk_stride":       stride,
        "visual_stride":      visual_stride,
        "num_video_frames":   num_video_frames,
    }


def run_phase2_vision(
    args, my_episodes: List[Dict], dev, dt,
    rank: int, local_rank: int, node_rank: int,
):
    """Phase 2: encode RGB+flow latents per episode, merge into same .pt.

    See ``run_phase1_text`` for the ``local_rank`` / ``node_rank`` rationale.
    """
    expected_stride = max(1, int(getattr(args, "chunk_stride", 1)))
    expected_vs = max(1, int(getattr(args, "visual_stride", 1)))
    _envf = getattr(args, "num_video_frames", None)
    expected_nvf = int(_envf) if _envf else int(args.num_frames)
    pending = [
        ep for ep in my_episodes
        if not _phase2_done(_pt_path_for(args.cache_root, ep), expected_stride,
                            expected_vs, expected_nvf)
    ]
    total = len(my_episodes)
    skipped = total - len(pending)
    if local_rank == 0:
        print(f"[Phase2][node{node_rank}] {skipped}/{total} episodes already "
              f"have VAE+flow on rank {rank}; {len(pending)} pending")

    if not pending:
        if local_rank == 0:
            print(f"[Phase2][node{node_rank}] Nothing to do.")
        return

    print(f"[Phase2] Rank {rank}: Loading VAE from {args.vae_path} ...")
    mm_vis = ModelManager()
    mm_vis.load_model(args.vae_path, device=dev, torch_dtype=dt)
    vae = mm_vis.fetch_model("wan_video_vae")
    assert vae is not None, "Failed to load VAE"

    # ------------------------------------------------------------------
    #  torch.compile the VAE encoder (Inductor fusion)
    # ------------------------------------------------------------------
    # We compile the inner nn.Module ``vae.model.encoder`` with the default
    # Inductor backend. ``mode="default"`` is chosen because Wan VAE's
    # temporal ``feat_cache`` uses string/tensor sentinel conditionals that
    # force graph breaks — ``reduce-overhead`` (CUDA graphs) cannot cross
    # those breaks, whereas default mode tolerates them and still fuses the
    # heavy conv/norm/silu chains. Typical steady-state speedup on A100 bf16
    # is ~1.1-1.2x; warm-up costs ~30s per rank. Compilation is correctness-
    # preserving modulo bf16 kernel noise, which is already captured by
    # PHASE2_VERSION so any old eager-encoded latents get invalidated and
    # re-encoded automatically.
    try:
        vae.model.encoder = torch.compile(
            vae.model.encoder, mode="default", fullgraph=False,
        )
        if local_rank == 0:
            print(f"[Phase2] Rank {rank}: VAE encoder compiled "
                  f"(mode=default, Inductor)")
    except Exception as exc:
        if local_rank == 0:
            print(f"[Phase2] Rank {rank}: torch.compile failed "
                  f"({type(exc).__name__}: {exc}); falling back to eager")

    print(f"[Phase2] Rank {rank}: Loading RAFT ...")
    raft_ext = RAFTFlowExtractor(device=str(dev))
    codec = FlowCodec()

    # ---- Compute target dimensions (with alignment) ----
    upf = vae.upsampling_factor
    h_div, w_div = upf * 2, upf * 2
    target_h = args.size_h + args.size_h // 2  # T-shape tiled height
    target_w = args.size_w
    if target_h % h_div != 0:
        target_h = (target_h + h_div - 1) // h_div * h_div
    if target_w % w_div != 0:
        target_w = (target_w + w_div - 1) // w_div * w_div
    print(f"[Phase2] Rank {rank}: target_w={target_w}, target_h={target_h}")

    pbar = tqdm(
        total=len(pending),
        desc=f"[node{node_rank} R{rank}] Phase2 VAE+flow",
        unit="ep", disable=(local_rank != 0), dynamic_ncols=True,
        position=0, leave=True,
    )
    t0 = time.time()
    try:
        for ep in pending:
            pt_path = _pt_path_for(args.cache_root, ep)
            pbar.set_postfix_str(f"{ep['task']}/{ep['variant']}/{ep['episode_name']}")

            try:
                result = process_episode_vision(
                    ep, args, vae, raft_ext, codec, dev, dt,
                    target_w, target_h,
                )
            except Exception as exc:
                print(f"[Phase2] Rank {rank}: FAILED "
                      f"{ep['task']}/{ep['variant']}/{ep['episode_name']}: {exc}")
                pbar.update(1)
                continue

            existing = _load_existing(pt_path)
            existing.update({
                "rgb_input_latents":  result["rgb_input_latents"],
                "flow_input_latents": result["flow_input_latents"],
                "start_indices":      result["start_indices"],
                "task":               ep["task"],
                "variant":            ep["variant"],
                "episode_name":       ep["episode_name"],
                "episode_length":     result["episode_length"],
                "num_chunks":         result["num_chunks"],
                "num_frames":         args.num_frames,
                "size_w":             args.size_w,
                "size_h":             args.size_h,
                "flow_mode":          args.flow_mode,
                "flow_max_magnitude": args.flow_max_magnitude,
                "chunk_stride":       result["chunk_stride"],
                "visual_stride":      result["visual_stride"],
                "num_video_frames":   result["num_video_frames"],
                "phase2_version":     PHASE2_VERSION,
            })
            # Fallback: if Phase 1 was skipped, still populate plain-text
            # `instructions` so older cache consumers keep working.
            if "instructions" not in existing:
                instr_file = os.path.join(
                    ep["variant_dir"], "instructions",
                    f"{ep['episode_name']}.json",
                )
                task_desc = ep["task"].replace("_", " ")
                instructions = [task_desc]
                if os.path.exists(instr_file):
                    try:
                        with open(instr_file, "r") as jf:
                            jdata = json.load(jf)
                        if "seen" in jdata and len(jdata["seen"]) > 0:
                            instructions = jdata["seen"]
                    except Exception:
                        pass
                existing["instructions"] = instructions

            _save_atomic(pt_path, existing)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pbar.update(1)
    finally:
        pbar.close()

    elapsed = time.time() - t0
    if local_rank == 0:
        print(f"[Phase2][node{node_rank}] Rank {rank}: {len(pending)} "
              f"episodes in {elapsed:.0f}s "
              f"({elapsed/max(len(pending),1):.1f}s/ep)")


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ---- Multi-GPU sharding ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    # torchrun sets GROUP_RANK == node_rank when launched with --nnodes > 1.
    # Fall back to rank//local_world_size for safety on older launchers.
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", max(1, world_size)))
    node_rank = int(os.environ.get("GROUP_RANK", rank // max(local_ws, 1)))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dt = torch.bfloat16

    if rank == 0:
        print(f"[Precompute] world_size={world_size}, cache_root={args.cache_root}")
        print(f"[Precompute] variants={args.variants}, flow_mode={args.flow_mode}")
        print(f"[Precompute] num_frames={args.num_frames}, size={args.size_w}x{args.size_h}")
        print(f"[Precompute] chunk_stride={args.chunk_stride} "
              f"(stride-mismatched Phase 2 files will be overwritten)")
        print(f"[Precompute] skip_phase1={args.skip_phase1}, skip_phase2={args.skip_phase2}")

    # ---- Discover episodes (across all variants) ----
    all_episodes = discover_all_episodes(
        args.dataset_base_path, args.variants, args.task_names, args.flow_mode,
    )
    if rank == 0:
        print(f"[Precompute] Total episodes: {len(all_episodes)} "
              f"(across {len(args.variants)} variants)")

    # ---- Shard episodes across ranks (load-balanced by trajectory length) ----
    # Per-episode work scales with trajectory length; round-robin (i % ws)
    # leaves ranks that drew long episodes (e.g. blocks_ranking) far behind,
    # and precompute finishes only when the SLOWEST rank does. LPT greedy
    # (longest-processing-time first) on hdf5 file size (∝ frame count) evens
    # out the total work per rank. Deterministic (same sorted input on every
    # rank) so each rank derives the identical split and takes its own bucket.
    # MUST be identical for Phase 1 and Phase 2 (they merge text/latent into the
    # same per-node .pt) — guaranteed since both consume this single split.
    def _ep_work(ep):
        try:
            return os.path.getsize(ep["data_hdf5"])
        except OSError:
            return 1
    _work = sorted(
        ((ep, _ep_work(ep)) for ep in all_episodes),
        key=lambda x: x[1], reverse=True,
    )
    _loads = [0] * world_size
    _buckets = [[] for _ in range(world_size)]
    for _ep, _w in _work:
        _r = min(range(world_size), key=lambda i: _loads[i])
        _buckets[_r].append(_ep)
        _loads[_r] += _w
    my_episodes = _buckets[rank]
    if rank == 0:
        _gb = [round(l / 1e9, 1) for l in _loads]
        print(f"[Precompute] LPT balance (GB/rank): min={min(_gb)} "
              f"max={max(_gb)} spread={round(max(_gb) - min(_gb), 1)}")
    print(f"[Precompute] Rank {rank} (node{node_rank}, local{local_rank}): "
          f"{len(my_episodes)} episodes assigned")

    if not my_episodes:
        print(f"[Precompute] Rank {rank}: Nothing to do.")
        return

    # ---- Phase 1: text ----
    if not args.skip_phase1:
        run_phase1_text(args, my_episodes, dev, dt, rank, local_rank, node_rank)
    elif local_rank == 0:
        print(f"[Precompute][node{node_rank}] --skip_phase1 set; "
              f"skipping T5 text encoding")

    # ---- Phase 2: VAE + RAFT ----
    if not args.skip_phase2:
        run_phase2_vision(args, my_episodes, dev, dt, rank, local_rank, node_rank)
    elif local_rank == 0:
        print(f"[Precompute][node{node_rank}] --skip_phase2 set; "
              f"skipping VAE+RAFT encoding")

    print(f"[Precompute] Rank {rank}: Done.")


if __name__ == "__main__":
    main()
