# Source and Runtime Versions

Audit date: 2026-07-24 (Asia/Shanghai)

This file records source provenance and declared runtime versions. It does not
claim that ignored environments, checkpoints, or model artifacts are
reproducible.

## Repository provenance

| Component | Location / URL | Immutable revision | Audit result |
| --- | --- | --- | --- |
| Unified integration | `/home/larry/kuavo_unified_stack`, branch `codex/unified-stack` | `96100b0992eb7bc6016be01486391532d42683c8` | HEAD remains based directly on `305f78a` plus the handoff commit; the worktree now contains the reviewed plans and new pinned submodules |
| Official Kuavo source | `/home/larry/kuavo_data_challenge`, `https://github.com/LejuRobotics/kuavo_data_challenge.git` | `305f78af1a073eb74c3f2f0424eed5ab4eb54759` | Clean; local `main` is 20 commits ahead of the remote `main` observed at `c0ca6a48047e1396e02ac1d83df2cd8b6ede17cd` |
| Classic source | `/home/larry/kuavo_ship_classic`, branch `ship/classic` | `f675e82d2bb39237976511f43b4cf2db22d2eb10` | Clean; 38 commits unique relative to `305f78a`; merge base is `fce8528303fe77d52bf1fc656ddefb1d7e145400` |
| LeRobot | `https://github.com/huggingface/lerobot.git` | `56b43cc88844cab4f231cf370a6c8eb8103bc9b8` | Proper submodule in the official source; uninitialized in the unified clone |
| OpenPI Kuavo fork | `/home/larry/openpi-kuavo`, branch `kuavo-lerobot-v3` | `7f8010a1abc01f9bdf5cdff1f9f69a2df70d2a16` | Clean; one commit on `15a9616a00943ada6c20a0f158e3adb39df2ccac`; published as `Larryi/openpi:kuavo-lerobot-v3` |
| LingBot-VLA v1 | `https://github.com/robbyant/lingbot-vla.git`; local copy at `/home/larry/kuavo_data_challenge/third_party/lingbot-vla` | `4eb34b7693a0565c67433f8fac9c59a2e67eb60b` | Local non-Git directory matches this upstream tree, excluding generated `*.egg-info` and empty submodule directories |
| LingBot-Depth submodule | `https://github.com/Robbyant/lingbot-depth` | `dd0f91328f21145951a6a4a376abf80a126a9f40` | Gitlink recorded by LingBot-v1; local directory is empty |
| MoGe submodule | `https://github.com/microsoft/MoGe.git` | `07444410f1e33f402353b99d6ccd26bd31e469e8` | Gitlink recorded by LingBot-v1; local directory is empty |
| LingBot-VLA v2 | `https://github.com/Larryi/lingbot-vla-v2.git`, branch `codex/kuavo-target-slots` | `d34898da7170a5bcbb094aec9ea99d5d299cf18c`, based on upstream `2838c1862bbec1ea47942fb61512130f635eb595` | Upstream did not implement sparse `target_start` placement. The published Kuavo patch series tests compact single-arm, sparse right-arm, and concatenated bimanual mappings in the existing `kdc_vla` environment. |

## Declared runtime matrix

| Runtime | Python | CUDA / JAX | Torch | Transformers | NumPy | LeRobot / LingBot |
| --- | --- | --- | --- | --- | --- | --- |
| ROS / classic tracked files | ROS Noetic on Ubuntu 20.04 implies system Python 3.8; pinned LeRobot requires Python >=3.10, so these must remain separate environments | CUDA 12 wheels are present; tracked requirements pin CUDA runtime `12.6.77`; no JAX | `2.7.1` | Not pinned by `requirements_total.txt` | `2.2.6` | LeRobot `56b43cc8` (`0.4.3` source metadata) |
| LingBot-v1 upstream | `3.12.3` in upstream README | CUDA `12.8`; no JAX | `2.8.0` | `4.51.3` | `1.26.4` | LingBot-v1 `4eb34b7`; upstream README separately asks for LeRobot `0cf8648`, which differs from the unified pin |
| LingBot-v2 fork | Declares Python >=3.8; exact deployment interpreter is not pinned | CUDA version not pinned in the audited tracked files; no JAX | `2.8.0` | `4.57.3` | `1.26.4` | LingBot-v2 `d34898d`, based on upstream `2838c18` |
| OpenPI Kuavo | >=3.11; project workflows freeze Python 3.11 | `jax[cuda12]==0.5.3`; A100 workflow uses CUDA NVCC `12.6.85`, Blackwell uses `12.8.93` | `2.7.1` | `4.53.2` | >=1.22.4,<2.0.0 | LeRobot `56b43cc8`; OpenPI `7f8010a` |

## Remote protection status

- The unified repository has fetch remotes `source` (local official worktree) and
  `upstream` (official GitHub). Both push URLs are `DISABLED`.
- Unified remote `personal` fetches from
  `https://github.com/Larryi/kuavo-unified.git` and pushes over SSH to
  `git@github.com:Larryi/kuavo-unified.git`.
- The classic source is available locally as the read-only tracking ref
  `source/ship/classic@f675e82`.
- OpenPI remote `fork` fetches from `https://github.com/Larryi/openpi.git`,
  pushes over SSH, and now contains branch `kuavo-lerobot-v3@7f8010a`.
- The official and classic source worktrees still name the official Kuavo URL as
  both fetch and push `origin`. Treat them as read-only even though Git does not
  enforce that.
- OpenPI `kuavo-lerobot-v3` was pushed only to the selected personal fork.
  A local read-only classic tracking ref was fetched. No merge or cherry-pick
  was performed, and the unified branch itself has not been pushed.

## Reproducibility blockers

1. LeRobot, LingBot-v1, patched LingBot-v2, and OpenPI are now pinned top-level
   submodules. LingBot-v1's nested gitlinks are pinned by its source commit but
   need not be initialized in every checkout.
2. Keep packed environments, checkpoints, credentials, caches, `outputs/`, and
   `artifacts/` outside Git.
