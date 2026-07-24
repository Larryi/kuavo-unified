# LingBot-VLA v2 Kuavo integration

## Action contract

- The model head remains 55-D to match pretraining.
- `kuavo_v2` maps a single-arm dataset as 7 relative joint positions plus one absolute gripper value in the left/default slots.
- `kuavo_v2_right_arm` maps a Task1-style right-arm-only Kuavo dataset into the right-arm slots: arm joints `7..13` and gripper `1`.
- `kuavo_v2_bimanual` maps both arms as 14 relative joint positions plus two absolute gripper values.
- `FeatureTransform.unapply()` converts relative joint predictions back to absolute joint targets before Kuavo executes them.

Do not add the current state to V2 actions again in `KuavoBaseRosEnv`.

## Environment

```bash
conda activate kdc_vla2
pip install -r requirements-lingbot-v2-extra.txt
```

Do not install the unrelated PyPI package `utils3d==0.1.1`. MoGe requires the
GitHub implementation pinned in `requirements-lingbot-v2-extra.txt`. Repair an
environment that previously installed the wrong package with:

```bash
python -m pip uninstall -y utils3d
python -m pip install --no-deps \
  'git+https://github.com/EasternJournalist/utils3d.git@3fab839f0be9931dac7c8488eb0e1600c236e183'
```

The supported upstream versions are defined by `/home/larry/lingbot-vla-v2/requirements.txt`.

## Normalization

Task1/single arm:

```bash
conda activate kdc_vla2
LINGBOT_V2_ROOT=/home/larry/lingbot-vla-v2 \
  tools/compute_lingbot_v2_norm_stats.sh \
  /mnt/pqssd/Real_PQ_3.0/TASK1_SZ/task1_supertrimmed_200 \
  assets/norm_stats/kuavo_v2_right_arm_meanstd.json
```

The committed Task1 config uses `kuavo_v2_right_arm` with per-joint MeanStd after applying relQpos. For Task2, copy the norm config, use `data_name=kuavo_v2_bimanual`, include all three cameras, and write a separate bimanual norm file.

## Training

Dry-run the composed torchrun command:

```bash
conda activate kdc_vla2
python kuavo_train/train_lingbot_v2.py policy.dry_run=true
```

Start Task1 training:

```bash
conda activate kdc_vla2
MPLCONFIGDIR=/tmp/matplotlib \
python kuavo_train/train_lingbot_v2.py
```

Defaults:

- micro batch 1, gradient accumulation 8;
- rank 8, alpha 16 LoRA;
- MSE flow-matching action loss (`loss_type=fm`);
- cosine LR from `5e-5` with 2% warmup;
- MoGe/LingBot-Depth and DINO-video teacher distillation enabled;
- Qwen attention/MLP and visual `qkv`/MLP use LoRA;
- action/state projections, alignment heads, routers and correction biases remain fully trainable;
- fused routed-expert tensors remain frozen because PEFT Linear LoRA cannot represent packed expert parameters.

The DCP checkpoint is resumable. The generated `hf_ckpt` is merged and contains no PEFT adapter keys.

## Deployment

Use `policy_type: lingbot_v2` and start from `configs/deploy/kuavo_env.lingbot_v2.yaml`.

Ship these files as real files:

- merged V2 `hf_ckpt`;
- Qwen3-VL tokenizer/config/processor files;
- the matching MeanStd JSON;
- `configs/robot_configs/kuavo_v2_right_arm.yaml`, `kuavo_v2.yaml`, or `kuavo_v2_bimanual.yaml`;
- the LingBot-VLA v2 Python package and runtime environment.

Do not ship MoGe, LingBot-Depth or DINO-video teacher checkpoints. They are training-only.
