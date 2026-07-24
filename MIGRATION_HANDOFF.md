# Kuavo Unified Stack Migration Handoff

## Purpose

This clone is the safe integration workspace for unifying:

- official Kuavo ROS, `KuavoBaseRosEnv`, evaluation, and shipping code;
- classic LeRobot ACT and Diffusion Policy training/inference;
- LingBot-VLA v1 and v2 training/inference;
- the Kuavo-compatible Physical Intelligence OpenPI JAX Pi0.5 fork;
- shared LeRobot v3 dataset mixing and cloud-training orchestration.

The source repositories remain untouched after the baseline commits listed below. Continue migration in this
directory rather than making broad integration changes in the original worktrees.

## Workspace identity

```text
Path:       /home/larry/kuavo_unified_stack
Branch:     codex/unified-stack
Baseline:   kuavo_data_challenge main@305f78a
Upstream:   https://github.com/LejuRobotics/kuavo_data_challenge.git
Source:     /home/larry/kuavo_data_challenge
```

`source` is a local read-only reference for comparison. `upstream` is the official project. Add a personal
Git remote later before pushing; do not push the integration branch to the official remote by accident.

This directory was created with:

```bash
git clone --local --no-hardlinks \
  /home/larry/kuavo_data_challenge \
  /home/larry/kuavo_unified_stack
```

Consequently, ignored model weights, `outputs/`, `artifacts/`, packed Conda environments, Docker image exports,
and local caches were not copied.

## Preserved source commits

All three source worktrees were clean when this handoff was created.

### Official main / LingBot v2 baseline

Repository: `/home/larry/kuavo_data_challenge`

```text
305f78a feat(lingbot-v2): harden Kuavo training and deployment
```

This commit contains right-arm and bimanual LingBot v2 configuration, normalization assets, dependency
compatibility checks, diagnostics, gripper latching, and deployment/open-loop safeguards.

### Classic branch commits to migrate

Repository: `/home/larry/kuavo_ship_classic`

Branch: `ship/classic`

```text
37aa243 fix(train): correct Task2 Google Drive downloads
9b13f56 feat(data): add Beijing dataset audit and export tools
128f896 feat(lingbot): add full Task1 cloud training pipeline
f675e82 docs: add robotics interview deep dives
```

The branch also contains earlier classic training, Docker, diagnostics, policy-loader, ACT/DP, action
postprocessing, dataset QA, and cloud pipeline commits not present on `main`. Do not merge the entire branch
blindly. Migrate by capability after comparing each overlapping deploy/config file.

### OpenPI fork baseline

Repository: `/home/larry/openpi-kuavo`

Branch: `kuavo-lerobot-v3`

```text
7f8010a feat(robots): add LeRobot v3 training workflows
```

This commit contains:

- LeRobot v3 at commit `56b43cc88844cab4f231cf370a6c8eb8103bc9b8`;
- local-root loading;
- Kuavo Task1/Task2 transforms;
- weighted multi-root and episode-subset mixtures;
- SO-101 support;
- JAX Pi0.5 training, checkpoint, open-loop, and policy-server smoke tests;
- Vast.ai training/upload/notification/shutdown tooling.

The OpenPI repository currently points `origin` at the Physical Intelligence upstream proxy. Before adding it
as a submodule here, create or select a durable personal fork remote and push `7f8010a`. Do not create a
submodule whose only reachable commit exists on one local machine.

## Important source reproducibility gaps

The original official worktree has:

- `third_party/lerobot` as a proper submodule pinned to `56b43cc8`;
- a local `third_party/lingbot-vla` directory ignored by Git;
- configs referencing `third_party/lingbot-vla-v2`, but no local v2 source directory;
- `myenv.tar.gz` (about 4.5 GB) and `myenv_lingbot.tar.gz` (about 6.8 GB), both ignored;
- exported Docker/model archives up to roughly 26 GB.

Before claiming reproducible LingBot builds, record v1 and v2 as pinned submodules or as checksummed download
artifacts. Do not commit packed environments or model weights to Git.

## Architectural decision

Unify the repository and operator workflow, not Python `site-packages`.

The required runtime model is:

```text
ROS / KuavoBaseRosEnv process
    -> lightweight policy client
    -> local model worker in an isolated environment
    -> action chunk
    -> centralized safety and action execution
    -> robot
```

Reasons:

- ROS Noetic/Focal normally uses Python 3.8.
- OpenPI requires Python 3.11, JAX CUDA, NumPy < 2, and its own lockfile.
- classic LeRobot currently uses a different Torch/NumPy stack.
- LingBot v1 and v2 have conflicting model/dependency packages and already require process isolation.

OpenPI's msgpack WebSocket `infer/reset` protocol is the preferred starting point. Classic and LingBot
workers should implement compatible servers rather than forcing all models into the ROS process.

## Canonical policy contract

Define this contract before migrating model loaders:

```python
reset() -> None

infer({
    "observation.images.head_cam_h": uint8[H, W, 3],
    "observation.images.wrist_cam_l": uint8[H, W, 3] | absent,
    "observation.images.wrist_cam_r": uint8[H, W, 3] | absent,
    "observation.state": float32[D],
    "prompt": str,
}) -> {
    "actions": float32[T, D],
}
```

Every deployable artifact must also declare:

- backend and checkpoint format;
- action semantics: absolute joint target, delta joint target, or EEF;
- joint/action names and dimensions;
- gripper units/range;
- action horizon and collection FPS;
- number of chunk steps executed before replanning;
- camera keys;
- normalization asset identity;
- source code commit and dataset revisions.

The ROS-facing controller should receive one canonical physical command representation. Delta-to-absolute
conversion, normalization, and gripper scaling must have exactly one owner to prevent double transforms.

## Proposed repository layout

```text
kuavo_unified_stack/
├── kuavo_deploy/                  # authoritative ROS/BaseEnv and controller
├── kuavo_train/
│   ├── classic/
│   ├── lingbot_v1/
│   ├── lingbot_v2/
│   └── launchers/
├── packages/
│   ├── kuavo_policy_protocol/
│   └── kuavo_lerobot_mix/
├── third_party/
│   ├── lerobot/
│   ├── openpi-kuavo/
│   ├── lingbot-vla/
│   └── lingbot-vla-v2/
├── configs/
│   ├── tasks/
│   ├── train/
│   └── deploy/
├── envs/
│   ├── openpi/
│   ├── classic/
│   ├── lingbot-v1/
│   └── lingbot-v2/
├── docker/
│   ├── base/
│   ├── runtime-openpi/
│   ├── runtime-classic/
│   ├── runtime-lingbot-v1/
│   └── runtime-lingbot-v2/
└── scripts/
    └── kuavo
```

Treat this as a target, not a mandate to move every existing file immediately. Preserve upstream-compatible
paths until each migration step has tests.

## Phase 1 migration sequence

### 1. Protect remotes and provenance

1. Add a personal writable remote for this integration repository.
2. Add a personal OpenPI fork and push commit `7f8010a`.
3. Record LingBot v1/v2 source URLs and immutable revisions.
4. Add a small `VERSIONS.md` with Python, CUDA, JAX, Torch, Transformers, LeRobot, and LingBot revisions.

### 2. Converge the official branches

Use `main@305f78a` as the base. Migrate from `ship/classic` in this order:

1. shared deployment utilities and action postprocessing;
2. ACT/DP loaders and training configs;
3. classic Docker/runtime build;
4. cloud training scripts;
5. Beijing data tools;
6. LingBot v1 cloud pipeline;
7. optional interview documentation.

Prefer cherry-picking commits that do not overlap. For overlapping files such as:

```text
Dockerfile
configs/deploy/kuavo_env.yaml
kuavo_deploy/config.py
kuavo_deploy/src/eval/real_single_test.py
kuavo_deploy/src/eval/sim_auto_test.py
kuavo_deploy/utils/policy_loader.py
```

reconcile behavior deliberately instead of accepting either branch wholesale.

### 3. Establish the protocol package

Create a dependency-light package containing:

- observation/action schema validation;
- backend metadata and artifact manifest;
- local policy client;
- reset, timeout, health-check, and error semantics;
- numpy/msgpack serialization tests.

Keep ROS and model-framework imports out of this package.

### 4. Extract shared LeRobot mixture support

Move the generic pieces of OpenPI's weighted mixture into `packages/kuavo_lerobot_mix`:

- source `repo_id/root/episodes/weight`;
- virtual concatenation;
- per-frame `source_weight / source_length`;
- deterministic single-process sampler;
- rank-aware DDP sampler with `set_epoch`;
- the same distribution for normalization stats.

OpenPI-specific transforms and DataConfig stay inside OpenPI. LingBot/ACT/DP should adopt only the generic
dataset/sampler layer.

### 5. Add isolated model workers

Implement and test one worker at a time:

1. classic ACT/DP;
2. LingBot v1;
3. LingBot v2;
4. OpenPI Pi0.5.

Each worker must pass the same offline fixture:

- load checkpoint;
- accept canonical observation;
- return finite `[horizon, action_dim]`;
- reset chunk/cache state;
- declare action semantics and normalization metadata.

### 6. Add task and artifact manifests

Task manifests choose datasets and robot mappings. Artifact manifests choose runtime backend and checkpoint.
Do not make a single YAML contain every backend's internal hyperparameters; reference native backend configs.

### 7. Build Docker targets

Use a shared ROS Noetic runtime base and backend-specific targets. The default competition path should build
only the backend selected by the artifact manifest.

If the competition permits one image per task, produce small task-specific images. If it requires one image
for all tasks, keep separate venvs and model worker processes inside that image; do not merge dependencies.

Model checkpoints should enter the build through an explicit artifact directory or BuildKit named context,
not through `COPY . .`.

## Phase 1 acceptance gates

Do not proceed to real-robot changes until:

1. repository clone plus documented artifact downloads reproduces all source trees;
2. no model weights, packed environments, credentials, or outputs are tracked by Git;
3. the canonical protocol has unit tests;
4. classic and OpenPI workers pass the same offline observation fixture;
5. ROS mock tests cover timeout, reset, disconnect, and malformed action shapes;
6. every action path declares absolute/delta/EEF semantics;
7. each Docker target has an import smoke test and one offline inference test;
8. upstream updates can be fetched without rewriting local integration history.

## First task for the next Codex project

Start in:

```text
/home/larry/kuavo_unified_stack
```

Recommended first request:

```text
Read MIGRATION_HANDOFF.md. Perform Phase 1 steps 1 and 2 only:
audit remotes and source revisions, then prepare a commit-by-commit migration plan from source/ship/classic
onto codex/unified-stack. Do not merge or cherry-pick until the overlap plan is reviewed.
```

This intentionally begins with provenance and conflict planning rather than immediate code movement.
