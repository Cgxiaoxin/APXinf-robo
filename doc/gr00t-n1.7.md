# GR00T N1.7 on LIBERO

Install APXinf-robo with its LIBERO dependencies, the LIBERO simulator, the
ApxInf CUDA binding, and
the matching Isaac-GR00T/Transformers processor environment. Prepare the
checkpoint's local Cosmos processor resources as described in the
[ApxInf loading guide](../apxinf/doc/gr00t-n1.7.md#loading).

## Performance

Batch 1, best recorded model-core P50 from fixed processor tensors to returned
actions, following the [GR00T benchmark procedure](../apxinf/doc/gr00t-n1.7.md#fixed-input-benchmark):

| Hardware | Precision | 1-view P50 | 2-view P50 |
|---|---|---:|---:|
| Jetson AGX Thor | BF16 | 51.834 ms | 54.216 ms |
| Jetson AGX Thor | FP8 | 32.557 ms | 35.436 ms |
| Jetson AGX Orin | BF16 | 75.778 ms | 84.864 ms |
| Jetson AGX Orin | W8A8 | 56.711 ms | 64.924 ms |

Download the official
[NVIDIA GR00T-N1.7-LIBERO checkpoint](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO/tree/main/libero_10)
and the
[LIBERO-10 dataset](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot)
referenced by NVIDIA's
[LIBERO guide](https://github.com/NVIDIA/Isaac-GR00T/blob/51d4c89f72fda44cbf77285c6a8114b52676b8a1/examples/LIBERO/README.md):

```sh
hf download nvidia/GR00T-N1.7-LIBERO \
  --revision 2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21 \
  --include "libero_10/*" --local-dir /models/GR00T-N1.7-LIBERO
hf download --repo-type dataset \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
  --revision e1a223d30b896c1613f270a2bfc63d382b3de7e1 \
  --local-dir /data/libero_10_no_noops_1.0.0_lerobot
```

Use NVIDIA Isaac-GR00T revision
[`51d4c89`](https://github.com/NVIDIA/Isaac-GR00T/tree/51d4c89f72fda44cbf77285c6a8114b52676b8a1),
copy its LIBERO modality description into the downloaded dataset, and run the
official processor to create both fixed-input fixtures:

```sh
git clone https://github.com/NVIDIA/Isaac-GR00T.git /opt/Isaac-GR00T
git -C /opt/Isaac-GR00T checkout 51d4c89f72fda44cbf77285c6a8114b52676b8a1
cp /opt/Isaac-GR00T/examples/LIBERO/modality.json \
  /data/libero_10_no_noops_1.0.0_lerobot/meta/modality.json
python scripts/prepare_gr00t_fixture.py \
  --checkpoint /models/GR00T-N1.7-LIBERO/libero_10 \
  --backbone /models/GR00T-N1.7-LIBERO/libero_10/assets/cosmos \
  --dataset /data/libero_10_no_noops_1.0.0_lerobot
```

This reads public episode 0, step 0 and writes `one-view/` and `two-view/`
under `devlocal/gr00t-n1d7/fixtures/`. The tensors are deterministic model-core
inputs with fixed-zero diffusion noise. Each manifest records both source-file
and generated-tensor SHA256 hashes. They reproduce the benchmark input shapes
and can also drive model-core numerical parity checks; they do not replace the
closed-loop LIBERO accuracy evaluation below.

Run the pinned model-core CUDA Graph benchmark with the generated fixture.
Choose the one- or two-view directory for the matching table column:

```sh
python scripts/bench_gr00t.py \
  --checkpoint /models/GR00T-N1.7-LIBERO/libero_10 \
  --backbone /models/GR00T-N1.7-LIBERO/libero_10/assets/cosmos \
  --fixture devlocal/gr00t-n1d7/fixtures/two-view --precision bf16 \
  --warmup 10 --iterations 50 \
  --output devlocal/gr00t-eval/latency.json
```

## Accuracy evaluation

LIBERO-10, two views, ten episodes per task:

| Hardware | Precision | Episodes | Successes | Success rate |
|---|---|---:|---:|---:|
| Jetson AGX Thor | BF16 | 100 | 94 | 94.0% |
| Jetson AGX Thor | FP8 | 100 | 92 | 92.0% |
| Jetson AGX Orin | BF16 | 100 | 93 | 93.0% |
| Jetson AGX Orin | W8A8 | 100 | 93 | 93.0% |

Run the full suite with Robo's GR00T state and action conversion:

```sh
apxinf-robo eval-libero --backend in-process \
  --model-dir /models/GR00T-N1.7-LIBERO/libero_10 --precision bf16 \
  --suite libero_10 --trials-per-task 10 --seed 7 \
  --max-steps 720 --replan-steps 8 \
  --results-jsonl devlocal/gr00t-eval/full-results.jsonl \
  --summary-json devlocal/gr00t-eval/full-summary.json
```

For a service deployment, use `apxinf-robo serve --robot franka_libero
--model-dir /models/GR00T-N1.7-LIBERO/libero_10 --precision bf16`, then select
`--backend websocket` in the evaluator. See the [observation and gripper
contract](../examples/README.md#gr00t-n17-on-libero).
