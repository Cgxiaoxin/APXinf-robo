#!/usr/bin/env python3
"""Build deterministic ApxInf GR00T benchmark fixtures from official LIBERO data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ISAAC_GROOT_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
DATASET_ID = "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot"
DATASET_REVISION = "e1a223d30b896c1613f270a2bfc63d382b3de7e1"
CHECKPOINT_ID = "nvidia/GR00T-N1.7-LIBERO"
CHECKPOINT_REVISION = "2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21"
SCHEMA = "apxinf.gr00t-n1.7.preprocessed-fixture.v1"
VIEW_KEYS = {
    "one-view": ["image"],
    "two-view": ["image", "wrist_image"],
}
EXPECTED_SHAPES = {
    "one-view": {
        "pixel_values": [256, 1536],
        "image_grid_thw": [1, 3],
        "token_ids": [1, 90],
        "attention_mask": [1, 90],
        "state": [1, 1, 132],
        "noise": [1, 40, 132],
    },
    "two-view": {
        "pixel_values": [512, 1536],
        "image_grid_thw": [2, 3],
        "token_ids": [1, 156],
        "attention_mask": [1, 156],
        "state": [1, 1, 132],
        "noise": [1, 40, 132],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="local nvidia/GR00T-N1.7-LIBERO/libero_10 checkpoint",
    )
    parser.add_argument(
        "--backbone",
        type=Path,
        required=True,
        help="local Cosmos-Reason2-2B processor/tokenizer directory",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=f"local {DATASET_ID} dataset with meta/modality.json installed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("devlocal/gr00t-n1d7/fixtures"),
        help="fixture parent directory (default: %(default)s)",
    )
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--step-index", type=int, default=0)
    parser.add_argument(
        "--views",
        choices=("one-view", "two-view", "both"),
        default="both",
        help="fixtures to generate (default: %(default)s)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing generated fixture directory",
    )
    return parser.parse_args()


def _require_inputs(args: argparse.Namespace) -> None:
    for label in ("checkpoint", "backbone", "dataset"):
        path = getattr(args, label)
        if not path.is_dir():
            raise SystemExit(f"--{label} is not a directory: {path}")
    modality = args.dataset / "meta/modality.json"
    if not modality.is_file():
        raise SystemExit(
            f"missing {modality}; copy examples/LIBERO/modality.json from the pinned "
            "Isaac-GR00T checkout into the downloaded dataset's meta directory"
        )
    if args.episode_index < 0 or args.step_index < 0:
        raise SystemExit("--episode-index and --step-index must be non-negative")


def _source_revision() -> str | None:
    try:
        import gr00t

        root = Path(gr00t.__file__).resolve().parent.parent
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (ImportError, OSError, subprocess.CalledProcessError):
        return None


def _load_processor_and_episode(args: argparse.Namespace):
    try:
        import gr00t.model  # noqa: F401 -- registers GR00T with Transformers
        from gr00t.data.embodiment_tags import EmbodimentTag
        from transformers import AutoProcessor
    except ImportError as error:
        raise SystemExit(
            "fixture generation requires the pinned NVIDIA Isaac-GR00T environment; "
            "follow the setup commands in doc/gr00t-n1.7.md"
        ) from error

    revision = _source_revision()
    if revision is not None and revision != ISAAC_GROOT_REVISION:
        raise SystemExit(
            "Isaac-GR00T revision mismatch: "
            f"found {revision}, expected {ISAAC_GROOT_REVISION}"
        )

    processor_dir = args.checkpoint
    nested_processor = args.checkpoint / "processor"
    if nested_processor.is_dir() and not (processor_dir / "processor_config.json").is_file():
        processor_dir = nested_processor
    processor = AutoProcessor.from_pretrained(
        processor_dir,
        model_name=str(args.backbone),
        transformers_loading_kwargs={"trust_remote_code": True, "local_files_only": True},
        local_files_only=True,
    )
    processor.eval()
    embodiment = EmbodimentTag.LIBERO_PANDA
    modality_configs = copy.deepcopy(processor.get_modality_configs()[embodiment.value])
    sample = _load_dataset_step(
        args.dataset,
        args.episode_index,
        args.step_index,
        modality_configs,
    )
    return processor, embodiment, sample, revision


def _load_dataset_step(
    dataset: Path,
    episode_index: int,
    step_index: int,
    modality_configs: Any,
) -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
    except ImportError as error:
        raise SystemExit(
            "fixture generation requires pandas, PyArrow, and NumPy"
        ) from error

    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    modality = json.loads((dataset / "meta/modality.json").read_text(encoding="utf-8"))
    episodes = [
        json.loads(line)
        for line in (dataset / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_index = {int(item["episode_index"]): item for item in episodes}
    if episode_index not in by_index:
        raise SystemExit(f"episode {episode_index} is not present in dataset metadata")
    episode_meta = by_index[episode_index]
    if step_index >= int(episode_meta["length"]):
        raise SystemExit(
            f"step {step_index} is outside episode range 0..{int(episode_meta['length']) - 1}"
        )
    chunk = episode_index // int(info["chunks_size"])
    format_values = {"episode_chunk": chunk, "episode_index": episode_index}
    parquet_path = dataset / info["data_path"].format(**format_values)
    frame_table = pd.read_parquet(parquet_path)
    if step_index >= len(frame_table):
        raise SystemExit(
            f"step {step_index} is outside parquet range 0..{len(frame_table) - 1}"
        )
    row = frame_table.iloc[step_index]

    states = {}
    for key in modality_configs["state"].modality_keys:
        descriptor = modality["state"][key]
        source = descriptor.get("original_key", "observation.state")
        states[key] = np.asarray(
            row[source][int(descriptor["start"]) : int(descriptor["end"])],
            dtype=np.float32,
        )[None, :]

    images = {}
    source_files = {str(parquet_path.relative_to(dataset)): _sha256_file(parquet_path)}
    for key in VIEW_KEYS["two-view"]:
        source = modality["video"][key]["original_key"]
        relative = info["video_path"].format(video_key=source, **format_values)
        video_path = dataset / relative
        images[key] = [_decode_frame(video_path, step_index)]
        source_files[str(video_path.relative_to(dataset))] = _sha256_file(video_path)

    tasks = episode_meta.get("tasks", [])
    if not tasks or not isinstance(tasks[0], str):
        raise SystemExit(f"episode {episode_index} has no language task")
    return {
        "images": images,
        "states": states,
        "prompt": tasks[0],
        "source_files": source_files,
    }


def _decode_frame(path: Path, frame_index: int):
    try:
        from PIL import Image
        from gr00t.utils.video_utils import get_frames_by_indices

        frames = get_frames_by_indices(str(path), [frame_index])
        return Image.fromarray(frames[0])
    except (ImportError, OSError, RuntimeError) as torchcodec_error:
        try:
            import av
            from PIL import Image
        except ImportError as error:
            raise SystemExit(
                "video decoding requires the pinned Isaac-GR00T TorchCodec setup "
                "or PyAV as a fallback"
            ) from error
        with av.open(str(path)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index == frame_index:
                    return Image.fromarray(frame.to_ndarray(format="rgb24"))
        raise SystemExit(f"video {path} has no frame {frame_index}") from torchcodec_error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collate_view(processor, embodiment, sample_data: dict[str, Any], name: str):
    try:
        from gr00t.data.types import MessageType, VLAStepData
    except ImportError as error:
        raise SystemExit("cannot import the pinned Isaac-GR00T data pipeline") from error

    selected_keys = VIEW_KEYS[name]
    sample = VLAStepData(
        images={key: sample_data["images"][key] for key in selected_keys},
        states=sample_data["states"],
        actions={},
        text=sample_data["prompt"],
        embodiment=embodiment,
    )

    processor_video = processor.modality_configs[embodiment.value]["video"]
    original_keys = processor_video.modality_keys
    processor_video.modality_keys = selected_keys
    try:
        processed = processor(
            [{"type": MessageType.EPISODE_STEP.value, "content": sample}]
        )
        collated = processor.collator([processed])
    finally:
        processor_video.modality_keys = original_keys
    return collated["inputs"] if "inputs" in collated else collated, sample.text


def _as_tensor(inputs: Any, name: str):
    if name not in inputs:
        raise SystemExit(f"official processor output is missing {name!r}")
    value = inputs[name]
    if not hasattr(value, "shape") or not hasattr(value, "detach"):
        raise SystemExit(f"official processor output {name!r} is not a tensor")
    return value.detach().cpu().contiguous()


def _write_tensor(path: Path, tensor, dtype: str) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        raise SystemExit("fixture serialization requires NumPy and PyTorch") from error

    shape = list(tensor.shape)
    if dtype == "bfloat16":
        array = tensor.to(torch.bfloat16).contiguous().view(torch.uint16).numpy().astype("<u2")
    elif dtype == "uint32":
        source = tensor.numpy()
        if np.any(source < 0) or np.any(source > np.iinfo(np.uint32).max):
            raise SystemExit(f"{path.name} contains a value outside uint32")
        array = source.astype("<u4")
    elif dtype == "uint8":
        source = tensor.numpy()
        if np.any(source < 0) or np.any(source > np.iinfo(np.uint8).max):
            raise SystemExit(f"{path.name} contains a value outside uint8")
        array = source.astype("u1")
    else:
        raise ValueError(f"unsupported fixture dtype {dtype}")
    payload = array.tobytes(order="C")
    path.write_bytes(payload)
    return {
        "bytes": len(payload),
        "dtype": dtype,
        "file": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "shape": shape,
    }


def _write_fixture(
    root: Path,
    name: str,
    inputs: Any,
    sample_data: dict[str, Any],
    processor: Any,
    args: argparse.Namespace,
    source_revision: str | None,
) -> None:
    try:
        import torch
    except ImportError as error:
        raise SystemExit("fixture generation requires PyTorch") from error

    tensors = {
        "pixel_values": (_as_tensor(inputs, "pixel_values"), "pixel_values.bf16", "bfloat16"),
        "image_grid_thw": (
            _as_tensor(inputs, "image_grid_thw"),
            "image_grid_thw.u32",
            "uint32",
        ),
        "token_ids": (_as_tensor(inputs, "input_ids"), "token_ids.u32", "uint32"),
        "attention_mask": (
            _as_tensor(inputs, "attention_mask"),
            "attention_mask.u8",
            "uint8",
        ),
        "state": (_as_tensor(inputs, "state"), "state.bf16", "bfloat16"),
        "noise": (
            torch.zeros(
                (1, int(processor.max_action_horizon), int(processor.max_action_dim)),
                dtype=torch.bfloat16,
            ),
            "initial_noise.bf16",
            "bfloat16",
        ),
    }
    actual_shapes = {key: list(value[0].shape) for key, value in tensors.items()}
    expected = EXPECTED_SHAPES[name]
    if actual_shapes != expected:
        raise SystemExit(
            f"{name} processor shapes do not match the published benchmark profile:\n"
            f"expected {expected}\nactual   {actual_shapes}"
        )

    embodiment = _as_tensor(inputs, "embodiment_id").reshape(-1)
    if embodiment.numel() != 1:
        raise SystemExit(f"expected one embodiment id, got shape {list(embodiment.shape)}")
    embodiment_id = int(embodiment[0].item())
    if embodiment_id != 2:
        raise SystemExit(f"expected LIBERO embodiment id 2, got {embodiment_id}")

    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists() and not args.force:
        raise SystemExit(f"fixture already exists: {root}; pass --force to replace it")
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}-", dir=root.parent))
    try:
        entries = {
            key: _write_tensor(temporary / filename, tensor, dtype)
            for key, (tensor, filename, dtype) in tensors.items()
        }
        manifest = {
            "schema": SCHEMA,
            "fixture": (
                f"official-libero-episode{args.episode_index}-"
                f"step{args.step_index}-{name}-v1"
            ),
            "embodiment_id": embodiment_id,
            "source": {
                "dataset": DATASET_ID,
                "dataset_url": f"https://huggingface.co/datasets/{DATASET_ID}",
                "dataset_revision": DATASET_REVISION,
                "checkpoint": CHECKPOINT_ID,
                "checkpoint_revision": CHECKPOINT_REVISION,
                "episode_index": args.episode_index,
                "step_index": args.step_index,
                "prompt": sample_data["prompt"],
                "views": VIEW_KEYS[name],
                "isaac_gr00t_revision": source_revision or ISAAC_GROOT_REVISION,
                "files": {
                    key: value
                    for key, value in sample_data["source_files"].items()
                    if "wrist_image" not in key or name == "two-view"
                },
            },
            "noise": {
                "kind": "fixed-zero",
                "reason": "deterministic cross-runtime comparison",
            },
            "tensors": entries,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if root.exists():
            shutil.rmtree(root)
        temporary.replace(root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    _require_inputs(args)
    processor, embodiment, sample, revision = _load_processor_and_episode(args)
    selected = VIEW_KEYS if args.views == "both" else {args.views: VIEW_KEYS[args.views]}
    for name in selected:
        inputs, _prompt = _collate_view(processor, embodiment, sample, name)
        destination = args.output / name
        _write_fixture(destination, name, inputs, sample, processor, args, revision)
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
