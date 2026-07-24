# Classic-to-Unified Commit Migration Plan

Status: review required. No merge or cherry-pick has been performed.

## Audited graph

- Target: `codex/unified-stack@96100b0`, whose parent is the official baseline
  `main@305f78a`.
- Source: `/home/larry/kuavo_ship_classic`, `ship/classic@f675e82`.
- Merge base: `fce8528303fe77d52bf1fc656ddefb1d7e145400`.
- Divergence: 38 classic-only commits and 22 official-main-only commits.
- `git cherry -v 305f78a f675e82` marks all 38 classic commits as patch-unique.
  This does not mean all 38 behaviors are missing: several official commits
  independently implemented or superseded the same capabilities.
- At the branch tips, `kuavo_deploy/utils/diagnostics.py` and
  `kuavo_train/wrapper/policy/config_loading.py` are byte-identical. Other
  high-risk deploy/config files have diverged.

The classic branch is now available under the stable read-only tracking ref
`source/ship/classic@f675e82`; neither source worktree was changed. Keep the
target branch based on `305f78a`; do not merge the classic branch.

## Resolution rules

1. Preserve the official `305f78a` LingBot-v2, client-policy, diagnostics,
   timeout, reset, and action-contract behavior as the target baseline.
2. For a `PORT` entry, reproduce only the named behavior in a new focused
   commit. Do not resolve by taking `--ours`, `--theirs`, or a whole classic
   file.
3. For a `PICK` entry, cherry-pick only after review and after its listed
   predecessors. Remove `.bak` files from any candidate patch.
4. Keep normalization, delta-to-absolute conversion, gripper scaling, and
   gripper latching single-owned. In particular, retain the official LingBot-v2
   rule that `FeatureTransform.unapply()` owns relative-to-absolute conversion.
5. After every deploy/config port, run syntax/config loading tests and ROS mocks.
   After every training port, run config composition plus a CPU/import smoke
   test. Cloud scripts get `bash -n` and dry-run validation.

Legend:

- `DROP`: intentionally do not migrate.
- `PICK`: expected to apply as a commit after approval; still verify.
- `PORT`: manually replay selected behavior on the official file.
- `SPLIT`: retain additive files/hunks and drop superseded portions.
- `DEFER`: outside the Phase 1 convergence scope or explicitly optional.

## Proposed execution order

The order follows the handoff's capability order rather than blindly replaying
the historical branch order. Source hashes remain the provenance for each new
commit.

### A. Provenance and harmless base data

| Source commit | Decision | Planned treatment |
| --- | --- | --- |
| `2a7cb3b` sync official commit | `DROP` | Do not replay an old upstream sync onto the newer `305f78a` baseline. |
| `08d2263` dataset task configs | `PICK` | Add the three `configs/data/Task*_SZ_Real.yaml` files; validate paths and schema against current dataset loading. |
| `71f0cbb` gitignore | `DROP` | Superseded by the current ignore policy. |
| `41d8974` gitignore | `DROP` | Superseded; fold any genuinely missing safe patterns into one later hygiene commit. |

### B. Shared deployment utilities and action postprocessing

Begin this stage with a unified-baseline cleanup commit: remove the existing
SmolVLA imports, loader/config choices, and deployment validation; rename the
SmolVLA-named open-loop scripts and documentation to backend-neutral names;
retain ACT, DP, LingBot-v1/v2, and an OpenPI client adapter behind lazy imports.

| Source commit | Decision | Planned treatment |
| --- | --- | --- |
| `4ecf0fe` SmolVLA support | `DROP` | SmolVLA will not be trained or deployed. Do not carry its loader, configuration, or dependencies into the final delivery. Preserve generic open-loop capability separately. |
| `6a98602` deploy YAML | `DROP` | Old sample values and `.bak` file are not authoritative; retain current YAML. |
| `a1e3b44` open-loop tool | `SPLIT` | Preserve and evolve this into a backend-neutral open-loop tool for ACT, DP, LingBot-v1/v2, and OpenPI. Remove SmolVLA-specific loading and keep current absolute-frame/raw-row behavior. |
| `559136f` EEF/camera YAML | `PORT` | Translate only still-valid EEF type and camera-key defaults into the current schema; reject hard-coded local paths. |
| `a81e68b` GR00T open loop | `DROP` | GR00T is not part of final training or delivery. |
| `06e90bc` deploy YAML | `DROP` | Later classic sample-state edit; current official config remains authoritative. |
| `577ee50` open-loop timeline | `PORT` | Compare with official `cd0a527`/`234c70c`; replay only viewer behavior demonstrably absent, with raw-row and absolute-frame semantics preserved. |
| `ae856c3` gripper latch/diagnostics | `SPLIT` | `diagnostics.py` is already identical and current eval files contain newer behavior. Keep official diagnostics; port only missing latch semantics/config, then add focused latch tests. Do not add a contentless `utils/__init__.py` merely to preserve the patch. |
| `2993c31` claw command range | `PORT` | Reapply the command-bound invariant to the current `KuavoBaseRosEnv.py`; verify current LingBot-v2 absolute-action path and both left/right claws. |
| `3a2612e` action postprocessing/viewer | `SPLIT` | Add `action_postprocessing.py` as a focused shared utility; manually adapt viewer integration to the current viewer instead of taking the classic file. Establish exactly one owner for delta/absolute and gripper transforms. |

Gate B: ROS mock tests cover ACT/DP/LingBot-v1/LingBot-v2/OpenPI client setup,
client timeout and reset, malformed action shapes, latch transitions, and claw
bounds.

### C. ACT/DP loaders, training configs, and auxiliary policy work

| Source commit | Decision | Planned treatment |
| --- | --- | --- |
| `8d30d3b` deploy configs | `SPLIT` | Add `dp_r1.yaml` and `dp_r2.yaml`; manually reconcile the two-line ACT config change with official local-config loading. |
| `795270e` config reading | `SPLIT` | `config_loading.py` is already identical. Retain it; port only wrapper/viewer call-site fixes not already provided by official `1702d2a`, with tests for checkpoints nested under run directories. |
| `c90015b` diffusion compile | `PORT` | The classic tip contains substantial compile/training changes while official main has LingBot-v2 additions. Rebase compile helpers and config keys onto the official trainers; preserve official code and test enabled/disabled/fallback modes. |
| `93c43af` auxiliary EEF diffusion | `DROP` | EEF diffusion is not part of final training or delivery. |
| `82782f2` accelerate multi-GPU | `PORT` | Reconcile launch and error handling with the current trainer rather than replacing it; exercise single-process and two-rank dry runs. |
| `6dfe942` Task2 H100 profile | `PICK` | Add the H100 YAML and documentation after `c90015b`; validate referenced keys and paths. |
| `8c8a5e2` final checkpoint | `PORT` | Preserve the official trainer and ensure `epochlast` saves a complete deployable policy plus processors exactly once. |
| `40450b5` unused DDP parameters | `PICK` | Apply after `6dfe942`; confirm it matches the reconciled `Accelerator`/DDP setting rather than masking genuinely unused modules. |

Gate C: ACT and diffusion configs compose, checkpoint discovery works, a tiny
train/save/reload cycle succeeds, and all action semantics are explicit.

### D. Classic Docker/runtime

| Source commit | Decision | Planned treatment |
| --- | --- | --- |
| `5c3d4ec` Docker fix | `SPLIT` | Reconstruct intended dependency/runtime changes on the current Dockerfile. Drop `Dockerfile.bak`. |
| `0c8222a` Docker/ignore hygiene | `SPLIT` | Consolidate only missing output/artifact exclusions. Keep source/submodule inputs required by current official and LingBot-v2 builds. |
| `1d1c9eb` num2words | `PORT` | Add only if an import audit proves the classic runtime needs it; pin the package in the backend-specific environment. |
| `085adde` Docker context exclusions | `SPLIT` | Fold safe missing patterns into the single hygiene commit; do not create a standalone historical commit. |
| `f0236e4` outputs exclusion fix | `SPLIT` | Same consolidated hygiene commit; verify spelling and BuildKit context behavior. |
| `22f7e3d` bundled LeRobot | `SPLIT` | Add `docker/build_classic.sh`; adapt Dockerfile changes to the initialized `third_party/lerobot@56b43cc8`. No `.bak`, weights, or packed environments enter the image context. |

Gate D: build context contains no credentials/outputs/checkpoints; classic image
imports the pinned LeRobot source and passes an offline loader smoke test.

### E. Cloud training scripts

| Source commit | Decision | Planned treatment |
| --- | --- | --- |
| `ccbd3c8` A100 pipeline | `PICK` | Add requirements, script, and docs after the training config ports; replace machine-specific defaults with required arguments or documented env vars. |
| `37aa243` Task2 downloads | `PICK` | Apply immediately after `ccbd3c8`; verify file IDs/checksums without downloading training artifacts into Git. |

Gate E: `bash -n`, `--help`/dry-run, failure-on-missing-secret, and artifact path
checks pass.

### F. Dataset editing, QC, and Beijing export

| Source commit | Decision | Planned treatment |
| --- | --- | --- |
| `760c9c4` episode editor | `PICK` | Add the editor as the base for later editor commits. |
| `e78a145` editor v10 | `PICK` | Apply after `760c9c4`; retain filename for provenance initially, but document the internal version mismatch. |
| `20f0c3f` multithread conversion | `SPLIT` | Manually port converter changes onto official `CvtRosbag2Lerobot.py`; drop `CvtRosbag2Lerobot.py.bak`. Preserve official chunk/timestamp fixes. |
| `bde4812` geometry annotations | `DROP` | These tools belong to the discarded EEF/geometry supervision path. |
| `9cae901` QC/rebuild tools | `PICK` | Apply after the editor base; run `--help`/import smoke tests and fixture-based header-alignment checks. |
| `9b13f56` Beijing audit/export | `PICK` | Apply after `9cae901`; verify script paths, non-destructive defaults, deterministic export metadata, and no embedded dataset paths in committed outputs. |

Gate F: fixture conversion preserves timestamps/action alignment, audit commands
are read-only, and exporters require an explicit output directory.

### G. LingBot-v1 cloud pipeline

| Source commit | Decision | Planned treatment |
| --- | --- | --- |
| `128f896` full Task1 LingBot pipeline | `SPLIT` | Add new config/docs/requirements/scripts/export tool. Keep identical compatibility files unchanged. Manually replay the trainer and norm-stat hunks on official `305f78a`, preserving later LingBot-v2 compatibility and safeguards. Pin LingBot-v1 to `4eb34b7` and its nested gitlinks before declaring the bundle reproducible. |

Gate G: bundle creation rejects missing/unpinned LingBot source, performs no
credential capture, and passes config composition plus checkpoint-export dry
runs.

### H. Optional documentation

| Source commit | Decision | Planned treatment |
| --- | --- | --- |
| `f675e82` interview deep dives | `DROP` | Explicitly excluded from the unified repository. |

## Expected conflict ownership

| Path | Authoritative base | Resolution |
| --- | --- | --- |
| `Dockerfile`, `.dockerignore`, `.gitignore` | Official `305f78a` plus current submodule/runtime requirements | Manually reconstruct classic backend additions; never copy `.bak` files or broaden build context to artifacts. |
| `configs/deploy/kuavo_env.yaml` | Official schema and LingBot-v2 action contract | Port individual typed keys/defaults; no wholesale YAML resolution. |
| `kuavo_deploy/config.py` | Official policy/client and validation model | Add only validated classic fields; keep client and LingBot-v2 policy types. |
| `real_single_test.py`, `sim_auto_test.py` | Official timeout/reset/client/diagnostics flow | Insert backend-neutral action/latch calls at one point after postprocessing and before execution. |
| `policy_loader.py` | Official loader | Keep only ACT, DP, LingBot-v1/v2, and OpenPI/client entry points; remove SmolVLA, GR00T, and EEF diffusion; avoid import-time optional dependency failures. |
| `KuavoBaseRosEnv.py` | Official physical command semantics | Port only claw bounds; do not reintroduce delta-to-absolute conversion for LingBot-v2. |
| `train_policy*.py` | Official trainer plus LingBot-v2 launch behavior | Manually layer compile, DDP, and final-save behavior with focused tests. |
| LingBot-v1 trainer/norm tools | Official `305f78a` versions | Replay cloud-specific hunks only; preserve official compatibility hardening. |
| Open-loop viewer/eval tools | Official absolute-frame/raw-row/LingBot-v2 behavior | Port independent panels/diagnostics; no whole-file choice. |

## Approval checkpoints

Review should explicitly decide:

Resolved decisions:

1. remove SmolVLA training/deployment support while preserving a generic
   multi-backend open-loop tool;
2. drop GR00T (`a81e68b`);
3. drop EEF diffusion and geometry supervision (`93c43af`, `bde4812`);
4. omit interview documentation (`f675e82`);
5. use `https://github.com/Larryi/kuavo-unified.git` and
   `https://github.com/Larryi/openpi.git`;
6. LingBot-v2 is pinned to `d34898d`, a tested patch series based on upstream
   `2838c18` that preserves compact and bimanual mappings while honoring sparse
   `target_start` placement.

Only after those decisions should migration commits be created. Each resulting
commit message should include `Source-Commit: <full hash>` so split/manual ports
remain traceable.
