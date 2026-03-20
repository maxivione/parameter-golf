---
description: Launch and run training on RunPod 8xH100
---

# RunPod 8×H100 Launch Workflow

## 1. Deploy Pod

Go to: https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th

- Select **8x H100 SXM** (this is the competition GPU)
- Enable SSH terminal access
- Deploy and wait for it to come online
- SSH in

## 2. Setup (run these on the pod)

```bash
cd /workspace
git clone https://github.com/maxivione/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024
```

## 3. Run Config A — Full Stack (our best shot)

```bash
RUN_ID=fullstack_v1 \
NUM_LAYERS=5 DEPTH_RECURRENCE=3 MODEL_DIM=640 \
NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 LORA_RANK=4 \
TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=524288 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_BETA2=0.95 \
MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=1.0 \
FAKE_QUANT_BITS=6 SWA_CHECKPOINTS=7 \
USE_NORMUON=1 SLIDING_WINDOW_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## 4. If Config A OOMs, try Config B (smaller dim)

```bash
RUN_ID=fullstack_v2 \
NUM_LAYERS=5 DEPTH_RECURRENCE=3 MODEL_DIM=512 \
NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=3 LORA_RANK=4 \
TRAIN_SEQ_LEN=1024 TRAIN_BATCH_TOKENS=524288 \
MATRIX_LR=0.02 SCALAR_LR=0.02 TIED_EMBED_LR=0.03 \
MUON_MOMENTUM=0.99 MUON_BETA2=0.95 \
MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3000 GRAD_CLIP_NORM=1.0 \
FAKE_QUANT_BITS=6 SWA_CHECKPOINTS=7 \
USE_NORMUON=1 SLIDING_WINDOW_STRIDE=64 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## 5. Check results

After training finishes, look for these lines in the output:
```
final_int8_zlib_roundtrip val_loss:X.XXXX val_bpb:X.XXXX
final_sliding_window_eval stride:64 val_loss:X.XXXX val_bpb:X.XXXX
Total submission size int8+zlib: XXXXX bytes
```

The sliding window val_bpb is your submission score.
Total size must be under 16,000,000 bytes.

## 6. Save logs

```bash
cp logs/*.txt /workspace/
```

Then download from RunPod before terminating the pod.

## Budget

- $25 ≈ 1 hour of 8×H100 SXM
- Each full run (train + eval) takes ~12-15 minutes
- That's 3-4 runs max. Don't waste runs on debugging.
- Test with ITERATIONS=100 first (~1 min) to catch errors before a full run.
