# π0.5：OpenPI FP32 训练权重 → APXInf 快速推理（RTX 5090 / Jetson Thor）

> **Audience:** 人读上手 + AI agent 无上下文部署。  
> **Scope:** OpenPI 侧用 FP32/BF16 训练出的 π0.5 checkpoint，转到 APXInf 在 **RTX 5090 (`sm_120`)** 或 **Jetson AGX Thor (`sm_110`)** 上做加速推理。  
> **Not in scope:** 新模型 port、kernel 开发、FP8 标定细节（见文末链接）。

---

## 0. 30 秒结论（先读这个）

| 目标 | 设备 | 推荐精度 | 怎么选 | 5090/Thor 参考延迟 |
|---|---|---|---|---|
| **线上最快推理** | RTX 5090 | **`bf16`** PreferGraph + tactics | `model_variant="bf16"` 或 Robo `--precision bf16` | ~**27 ms** |
| **线上最快推理** | Jetson Thor | **`fp8`**（有标定）或 **`bf16`** | `auto` / `fp8` / `bf16` | FP8 ~**41 ms**；BF16 ~**72 ms** |
| **数值基线 / 对齐 gold** | 5090（已测） | **`fp32`** PreferGraph（显式） | `model_variant="fp32"`；**`auto` 永不选 fp32** | ~**66 ms**（graph；eager ~85 ms） |
| Thor 上的 FP32 | Thor | 可跑，**无公开 SLA** | 显式 `fp32` | 未发布数字；部署仍优先 FP8/BF16 |

**训练 dtype ≠ 推理 dtype。** OpenPI 训练/导出可以是 float32 权重；APXInf 部署时把权重加载到所选 executor。**要最快，选 BF16（5090）或 FP8（Thor），不要默认 FP32。**

---

## 1. AI agent 速查卡（机器可读清单）

复制下面块即可当 checklist；每步失败则停，不要跳过。

```text
GOAL: Deploy OpenPI π0.5 SFT weights on APXInf for fast infer on 5090 or Thor.

ENV SPLIT (do not mix):
  - convert:  source .../activate_cgq.sh openpi-ref
  - infer:    source .../activate_cgq.sh apxinf-5090   # or Thor build env
  - libero:   separate libero-eval env (optional)

CHECKPOINT CONTRACT (APXInf expects OpenPI PyTorch export, NOT Orbax):
  <ckpt>/model.safetensors          # required
  <ckpt>/metadata.pt                # preferred
  <ckpt>/assets/<asset_id>/norm_stats.json

STEPS:
  1. Convert JAX Orbax -> PyTorch with --config_name matching train config
  2. Copy assets/ into output dir
  3. Rebuild/install apxinf_py with --features cuda ON THE TARGET GPU
     - 5090: APXINF_CUDA_ARCH=sm_120 (auto-detect OK)
     - Thor: APXINF_CUDA_ARCH=sm_110 (build on device / JetPack CUDA)
  4. Choose precision:
     - fastest 5090: model_variant=bf16
     - fastest Thor: model_variant=fp8_static (+ calibration) else bf16
     - numeric baseline: model_variant=fp32 (explicit only)
  5. Smoke load + one infer with matching action_dim/state_dim/num_views/image_keys/asset_id
  6. Optional: gold/latency scripts under devlocal/pi05-rtx5090/scripts/
  7. Serve: apxinf-robo serve --precision bf16|fp8|int8|auto
     FP32 serve: Python Pi05Policy(..., model_variant="fp32") + WebsocketPolicyServer
     (Robo --precision does NOT accept fp32 yet)

HARD RULES:
  - auto NEVER resolves to fp32
  - TF32 remains OFF by default for FP32 path
  - do not mix openpi-ref and apxinf-5090 Python stacks
  - RGB wire frames; match training camera order in image_keys
```

**关键路径（相对本仓库根 `APXinf-robo/`）：**

| 用途 | 路径 |
|---|---|
| 本文 | `doc/pi05-fp32-train-fast-infer-deploy.md` |
| SFT 转换笔记 | `apxinf/doc/pi05-sft-fp32-deploy.md` |
| 5090 FP32 gold/latency | `apxinf/doc/pi05-rtx5090-fp32-pathb.md` |
| Python 加载 | `apxinf/python/apxinf/apxinf/policies/impls/pi05.py` |
| Robo serve CLI | `src/apxinf_robo/cli/serve.py` |
| Robo precision→variant | `src/apxinf_robo/engine.py` (`_pi05_variant`) |
| 5090 latency 脚本 | `devlocal/pi05-rtx5090/scripts/compare_openpi_apxinf_latency.py` |
| OpenPI 转换脚本 | `/mnt/sdb/cgq/projects/openpi/examples/convert_jax_model_to_pytorch.py` |
| 本机 env 激活 | `/mnt/sdb/cgq/projects/env/activate_cgq.sh` |

---

## 2. 端到端流水线

```text
OpenPI train (JAX / Orbax, often float32 weights on disk)
        │
        ▼  openpi-ref + convert_jax_model_to_pytorch.py --config_name <TRAIN_CFG>
OpenPI PyTorch export
  model.safetensors + metadata.pt + assets/
        │
        ▼  apxinf-5090 或 Thor 上 maturin --features cuda
APXInf load
  ├─ bf16  ──► 5090 PreferGraph / Thor BF16     【部署默认·快】
  ├─ fp8   ──► Thor（需标定）                    【Thor 最快】
  └─ fp32  ──► 显式 model_variant=fp32           【数值基线·较慢】
        │
        ▼
Python infer / Websocket serve / LIBERO eval
```

---

## 3. 环境与构建

### 3.1 本机已有 env（推荐）

```bash
# 转换（OpenPI）
source /mnt/sdb/cgq/projects/env/activate_cgq.sh openpi-ref

# 推理（5090）
source /mnt/sdb/cgq/projects/env/activate_cgq.sh apxinf-5090
export CUDA_VISIBLE_DEVICES=0
# 可选: export CUDA_HOME=/usr/local/cuda-12.9
```

**不要**把 OpenPI / LIBERO-MuJoCo / APXInf 塞进同一个脆弱混装 env 做 bring-up。LIBERO 评估建议独立 `libero-eval`，通过 websocket 打到 APXInf 进程。

### 3.2 从源码构建 APXInf CUDA 绑定

在**目标 GPU 机器**上构建（跨机 AOT 需正确设 `APXINF_CUDA_ARCH`）：

```bash
cd /mnt/sdb/cgq/projects/APXinf-robo
# Thor 示例: export APXINF_CUDA_ARCH=sm_110
# 5090 通常可自动检测 sm_120

pip install maturin
CARGO_TARGET_DIR=target/wheel maturin build --release --features cuda --auditwheel skip \
  -m apxinf/crates/apxinf-py/Cargo.toml
pip install --force-reinstall target/wheel/wheels/apxinf_py-*.whl
pip install -e "./apxinf/python/apxinf[serving]" --config-settings editable_mode=strict
pip install -e ".[libero,serve]"
```

| 设备 | SM | tactics 目录（仓库内） | `auto` 默认 |
|---|---|---|---|
| RTX 5090 | 120 | `apxinf/configs/tuning/nvidia/rtx5090-sm120/` | **bf16**（即使有 FP8 scales 也不 auto→fp8） |
| Jetson Thor | 110 | `apxinf/configs/tuning/nvidia/thor-sm110/` | 有标定 → **fp8_static**，否则 bf16 |
| Thor-U | 101 | `thor-sm101/` | 同 Thor 族 |

FP32 路径：默认 PreferGraph **可 capture**（5090 实测 `execution_mode=graph`，P50 ~66 ms）；`APXINF_PI05_EXECUTION_POLICY=eager` 可强制 eager（~85 ms）。

---

## 4. Checkpoint：JAX → PyTorch

APXInf **不能**直接吃 Orbax `params/` 树。需要 OpenPI 导出布局：

```text
<pytorch_ckpt>/
  model.safetensors
  metadata.pt                 # openpi pytorch export，优先
  assets/<asset_id>/norm_stats.json
```

### 4.1 转换命令

`--config_name` **必须**与训练 config 一致（决定 π0.5 标志、`action_dim` 等）。脚本必填该参数。

```bash
source /mnt/sdb/cgq/projects/env/activate_cgq.sh openpi-ref
cd /mnt/sdb/cgq/projects/openpi

# 先确认 config（示例名请换成你的训练 config）
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir /mnt/sdb/cgq/projects/openpi/weight/21000 \
  --config_name <YOUR_TRAIN_CONFIG> \
  --output_path /mnt/sdb/cgq/projects/APXinf-robo/devlocal/ckpts/my_pi05_pytorch \
  --precision bfloat16
# 若你明确要导出 float32 权重文件，可改 --precision float32；
# APXInf 仍按 model_variant 决定设备上的 executor。

mkdir -p /mnt/sdb/cgq/projects/APXinf-robo/devlocal/ckpts/my_pi05_pytorch/assets
cp -a /mnt/sdb/cgq/projects/openpi/weight/21000/assets/. \
  /mnt/sdb/cgq/projects/APXinf-robo/devlocal/ckpts/my_pi05_pytorch/assets/
```

### 4.2 与训练对齐的旋钮

加载前确认（错一个就会 silent 坏结果或 shape 炸）：

| 旋钮 | 含义 | 例子 |
|---|---|---|
| `action_dim` | 部署动作维（可 trim） | LIBERO=7；某 SFT=16 |
| `state_dim` | `discrete_state=True` 时 proprio 宽 | 常与训练 proprio 一致 |
| `num_views` / `image_keys` | 相机数与 **训练顺序** wire key | `("base_0_rgb","left_wrist_0_rgb")` |
| `asset_id` | `assets/<id>/norm_stats.json` | 与训练 asset 名一致 |
| `discrete_state` | 是否把 state 离散进 prompt | 跟 OpenPI 训练一致 |
| `num_flow_steps` | flow 步数 | 常用 10 |

示例：`tianji_pi05_16d_direct` → `action_dim=16`，不是 DROID 的 8。

---

## 5. 快速加载与推理

### 5.1 最快路径（5090：BF16）

```bash
source /mnt/sdb/cgq/projects/env/activate_cgq.sh apxinf-5090
export CUDA_VISIBLE_DEVICES=0

python - <<'PY'
from apxinf import Pi05Policy
import numpy as np

CKPT = "/mnt/sdb/cgq/projects/APXinf-robo/devlocal/ckpts/my_pi05_pytorch"

policy = Pi05Policy.from_pretrained(
    CKPT,
    device="cuda:0",
    model_variant="bf16",          # 5090 部署默认
    discrete_state=True,           # 按你的训练改
    state_key="state",
    image_keys=("base_0_rgb", "left_wrist_0_rgb"),
    num_views=2,
    action_dim=16,
    state_dim=16,
    num_flow_steps=10,
    asset_id="tianji_pi05_16d_direct",  # 换成你的 asset
)

obs = {
    "base_0_rgb": np.zeros((224, 224, 3), np.uint8),
    "left_wrist_0_rgb": np.zeros((224, 224, 3), np.uint8),
    "state": np.zeros(16, np.float32),
    "prompt": "pick up the cup",
}
out = policy.infer(obs)
print(out["actions"].shape, out.get("timing"), policy.metadata.get("model_variant"))
policy.close()
PY
```

也可用 Robo 封装（精度拼写是 `precision`，内部映射到 `model_variant`）：

```python
from apxinf_robo import build_robot_policy
policy = build_robot_policy("franka_libero", CKPT, precision="bf16")
```

### 5.2 FP32 数值基线（显式）

```python
policy = Pi05Policy.from_pretrained(
    CKPT,
    device="cuda:0",
    model_variant="fp32",   # 必须显式；auto 不会选它
    # ... 其余与训练对齐 ...
)
```

### 5.3 Thor 最快路径

```python
# 有 FP8 标定文件时：
policy = Pi05Policy.from_pretrained(
    CKPT, device="cuda:0",
    model_variant="fp8_static",
    calibration="/path/to/pi05_fp8_calibration.json",
    # ...
)
# 无标定时用 bf16，或 model_variant="auto"
```

标定与 warm-start（STEP）见：`apxinf/doc/pi05-fp8-calibration.md`、`apxinf/doc/run_warmstart_with_onestep.md`。

---

## 6. Serve / 客户端

### 6.1 标准部署（BF16 / FP8 / INT8 / auto）

```bash
source /mnt/sdb/cgq/projects/env/activate_cgq.sh apxinf-5090   # Thor 上用对应 env

apxinf-robo serve --robot franka_libero \
  --model-dir /path/to/pytorch_ckpt \
  --precision bf16 \
  --port 8000
# Thor 最快: --precision fp8 并提供 calibration
```

OpenPI client：

```python
from openpi_client import websocket_client_policy
client = websocket_client_policy.WebsocketClientPolicy("127.0.0.1", 8000)
actions = client.infer(observation)["actions"]
```

### 6.2 FP32 serve（当前 CLI 缺口与绕过）

今日状态：

- `apxinf-robo serve --precision` 只接受 `auto|fp8|bf16|int8`
- `engine._pi05_variant` **没有** `fp32` 映射
- 运行时 **支持** `model_variant="fp32"`（Rust + Python `Pi05Policy`）

绕过：进程内加载 FP32 policy 再挂 websocket：

```bash
source /mnt/sdb/cgq/projects/env/activate_cgq.sh apxinf-5090

python - <<'PY'
from apxinf import Pi05Policy
from apxinf.serving import WebsocketPolicyServer

CKPT = "/path/to/pytorch_ckpt"
policy = Pi05Policy.from_pretrained(
    CKPT,
    device="cuda:0",
    model_variant="fp32",
    # 填齐与训练一致的 image_keys / state_key / dims / asset_id ...
    metadata={"protocol": "openpi.websocket_policy", "model_variant": "fp32"},
)
try:
    WebsocketPolicyServer(policy, "0.0.0.0", 8000).serve_forever()
finally:
    policy.close()
PY
```

若要给 AI 改 CLI：在 `src/apxinf_robo/engine.py` 的 `_pi05_variant` 与 `cli/serve.py` 的 `--precision` choices 增加 `fp32` → `fp32`。

---

## 7. 验证：延迟与数值

### 7.1 5090 参考数（`pi05_droid_pytorch`，gold seed7）

| 路径 | P50 | 角色 |
|---|---:|---|
| APXInf **bf16** PreferGraph | ~**27.3 ms** | **部署默认** |
| APXInf **fp32** PreferGraph / RequireGraph | ~**66.0 ms** | 数值基线（`execution_mode=graph`） |
| APXInf **fp32** Eager | ~**84.8 ms** | 同权重对照（`APXINF_PI05_EXECUTION_POLICY=eager`） |
| APXInf **fp32** reference MQA（加速前） | ~**91.3 ms** | 历史对照 |
| OpenPI JAX | ~**61.2 ms** | 参考 |

Gold soft gate：`max_abs≤0.05`，`cosine≥0.99`，`rel_l2≤0.1`。  
细节与产物：`apxinf/doc/pi05-rtx5090-fp32-pathb.md`，`devlocal/pi05-rtx5090/logs/fp32/`。

### 7.2 Thor 公开数（README；非 FP32）

| 精度 | P50 | 备注 |
|---|---:|---|
| BF16 | 72.45 ms | 基线 |
| FP8 | **41.16 ms** | 最快常规路径 |
| BF16 + STEP | 44.05 ms | warm-start |
| FP8 + STEP | **26.32 ms** | 最快 + 少步 |

### 7.3 延迟 smoke 命令

```bash
python /mnt/sdb/cgq/projects/APXinf-robo/devlocal/pi05-rtx5090/scripts/compare_openpi_apxinf_latency.py \
  --engine apxinf --model-variant bf16 --gpu 0 \
  --apxinf-checkpoint /path/to/pytorch_ckpt \
  --out-dir /tmp/bf16_latency

# FP32 基线
python .../compare_openpi_apxinf_latency.py \
  --engine apxinf --model-variant fp32 --gpu 0 \
  --apxinf-checkpoint /path/to/pytorch_ckpt \
  --out-dir /tmp/fp32_latency
```

随机权重纯 runtime（不验证任务）：

```bash
python apxinf/scripts/bench_pi05.py --random-weights --precision bf16 --layer l1 \
  --views 2 --token-count 10 --action-horizon 10 --num-flow-steps 10 \
  --warmup 10 --samples 100 --autotune
```

### 7.4 LIBERO（分 env）

```bash
# A: APXInf server
source /mnt/sdb/cgq/projects/env/activate_cgq.sh apxinf-5090
apxinf-robo serve --robot franka_libero \
  --model-dir /path/to/pytorch_ckpt --precision bf16 --port 8000

# B: libero-eval
python apxinf/scripts/eval_libero.py \
  --backend websocket --precision bf16 \
  --suite libero_10 --tasks 0 --trials-per-task 1 \
  --results-jsonl /tmp/libero_r.jsonl --summary-json /tmp/libero_s.json
```

`eval_libero.py` 已有 `--precision fp32` 的 in-process 映射；websocket 路径要求 server 侧已是对应精度。

---

## 8. 精度怎么选（决策树）

```text
需要线上最低延迟？
  ├─ 5090 ──► bf16 (+ PreferGraph / sm120 tactics)
  └─ Thor ──► 有 FP8 标定？
                ├─ yes ──► fp8_static（可再加 STEP）
                └─ no  ──► bf16

需要更紧的数值对齐 / FP32 executor 基线？
  └─ model_variant=fp32（显式；接受 ~2–3× 慢于 5090 bf16）

不要：
  - 指望 auto 选出 fp32
  - 默认打开 TF32（会破坏当前 gold）
  - 把 Orbax params/ 直接喂给 APXInf
```

---

## 9. 故障排查

| 现象 | 常见原因 | 处理 |
|---|---|---|
| 找不到权重 / layout 错 | 仍是 JAX `params/` | 先 convert + 拷 `assets/` |
| shape / action 维不对 | `--config_name` 或 `action_dim` 与训练不符 | 对齐 train config |
| 任务成功率崩、延迟正常 | `image_keys` 顺序错、BGR、norm/`asset_id` 错 | 按训练相机序与 RGB；核对 `norm_stats` |
| `unsupported PI0.5 precision 'fp32'` | Robo CLI 未映射 | 用 `Pi05Policy(..., model_variant="fp32")` |
| 5090 上 FP8 很慢 | 非 ship 路径 | 用 bf16 |
| Thor 构建失败 | 错 arch / 错 CUDA | 本机设 `APXINF_CUDA_ARCH=sm_110`，JetPack CUDA |
| PreferGraph 在 FP32 回退 | 异常（5090 应 capture） | 查 workspace / `execution_mode`；可用 `require_graph` 失败闭合；BF16 仍是最快部署路径 |

---

## 10. 新 SFT 落盘 checklist

1. [ ] 拿到训练 `config_name` 与 `assets/<asset_id>/`
2. [ ] Orbax → PyTorch（`openpi-ref`），拷贝 assets
3. [ ] 目标机装好 `apxinf_py`（`--features cuda`）
4. [ ] **BF16（5090）或 FP8/BF16（Thor）** smoke：`infer` 一次，看 `actions` shape 与 `timing`
5. [ ] （可选）FP32 smoke 作数值基线
6. [ ] （可选）gold / latency 脚本
7. [ ] `apxinf-robo serve` + 真机/仿真 client；LIBERO 用分 env websocket
8. [ ] 记录：设备、`model_variant`、P50、`action_dim`、`asset_id`、commit

---

## 11. 相关文档

| 文档 | 内容 |
|---|---|
| [`apxinf/doc/pi05-sft-fp32-deploy.md`](../apxinf/doc/pi05-sft-fp32-deploy.md) | SFT 转换与 5090 加载短注 |
| [`apxinf/doc/pi05-rtx5090-fp32-pathb.md`](../apxinf/doc/pi05-rtx5090-fp32-pathb.md) | FP32 executor gold + latency |
| [`apxinf/doc/pi05-rtx5090-phase5-6-closure.md`](../apxinf/doc/pi05-rtx5090-phase5-6-closure.md) | 5090 BF16 ship 收口 |
| [`apxinf/doc/pi05-cuda-regression.md`](../apxinf/doc/pi05-cuda-regression.md) | Thor 等 CUDA 回归数字 |
| [`doc/pi0-fast.md`](pi0-fast.md) | PI0-FAST（另一模型族） |
| 仓库根 [`README.md`](../README.md) | 构建、serve、公开性能表 |

---

## 12. 给 AI 的最小成功标准

部署任务视为完成，当且仅当：

1. Checkpoint 为带 `model.safetensors` 的 OpenPI PyTorch 目录（含可用 `norm_stats`）。
2. 在目标 GPU 上 `Pi05Policy.from_pretrained(..., model_variant=<chosen>)` 成功，`infer` 返回 `(H, action_dim)`。
3. **速度路径**选用设备推荐精度（5090→`bf16`，Thor→`fp8`/`bf16`），而不是误用 `fp32` 当“加速”。
4. （若用户要数值基线）额外跑通 `model_variant=fp32` 一次，并注明延迟更慢是预期。
5. 未混用 `openpi-ref` 与 `apxinf-5090` 解释器。
