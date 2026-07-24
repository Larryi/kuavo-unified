# Classic 到 Unified 的逐提交迁移计划

状态：**等待审核**。当前未执行 merge、cherry-pick、fetch 或 push。

## 一、审计结论

- 目标分支：`codex/unified-stack@96100b0`。
- 官方基线：`main@305f78a`，`96100b0` 仅在其上增加交接文档。
- 来源分支：`/home/larry/kuavo_ship_classic` 中的
  `ship/classic@f675e82`。
- 两条分支的共同祖先：
  `fce8528303fe77d52bf1fc656ddefb1d7e145400`。
- 相对共同祖先，classic 有 38 个独有提交，官方主线有 22 个独有提交。
- `git cherry -v 305f78a f675e82` 显示 38 个 classic 提交均不存在补丁等价提交。
  但这不表示所有功能都缺失：主线已经独立实现或增强了其中若干能力。
- 两个分支的最终版本中：
  - `kuavo_deploy/utils/diagnostics.py` 完全一致；
  - `kuavo_train/wrapper/policy/config_loading.py` 完全一致；
  - Docker、部署 YAML、policy loader、训练入口和真实/仿真评测文件均已明显分叉。

因此禁止整体合并 `ship/classic`，也禁止在冲突时整文件选择 `ours` 或
`theirs`。迁移必须按能力拆分，并在新提交中记录原始提交哈希。

## 二、统一处置规则

计划使用以下标记：

- `丢弃`：功能已过时、已被主线覆盖或不应进入统一仓库。
- `直接迁移`：审核后可按原提交迁移，但仍需验证。
- `手工重放`：基于主线当前文件重新实现指定行为。
- `拆分迁移`：只迁移提交中的新增文件或仍有效的部分。
- `延后`：不属于本轮主线收敛范围，等待单独决策。

所有手工或拆分迁移提交都应在提交说明中加入：

```text
Source-Commit: <完整 classic 提交哈希>
```

冲突处理遵守以下原则：

1. 以 `305f78a` 的 LingBot-v2、client policy、诊断、超时、reset 和动作契约为准。
2. 不得重新引入 LingBot-v2 的重复“相对动作转绝对动作”；该转换继续由
   `FeatureTransform.unapply()` 单独负责。
3. 归一化、delta-to-absolute、夹爪缩放和夹爪锁存各自只能有一个所有者。
4. `.bak`、模型权重、打包环境、输出、缓存和凭据不得进入 Git。
5. 每迁移一个部署提交，执行配置加载、语法检查和 ROS mock。
6. 每迁移一个训练提交，执行配置组合、import smoke test 和最小保存/重载测试。
7. 云端脚本至少通过 `bash -n`、`--help` 或 dry-run，并验证缺少凭据时安全失败。

## 三、建议迁移顺序

以下顺序按能力依赖排列，不机械沿用 classic 的历史顺序。

### A. 来源整理和基础数据配置

| 来源提交 | 处置 | 中文说明 |
| --- | --- | --- |
| `2a7cb3b` sync official commit | `丢弃` | 这是旧上游同步提交，不应重放到更新的 `305f78a` 基线上。 |
| `08d2263` dataset task configs | `直接迁移` | 增加三个 `configs/data/Task*_SZ_Real.yaml`；迁移后检查路径和当前数据配置 schema。 |
| `71f0cbb` gitignore | `丢弃` | 已被当前忽略规则覆盖。 |
| `41d8974` gitignore | `丢弃` | 不单独迁移；确实缺失的安全规则统一放入后续 hygiene 提交。 |

### B. 共享部署工具和动作后处理

本阶段首先增加一个统一仓库清理提交，移除官方基线中已经存在的
SmolVLA 部署支持：

- 从 `kuavo_deploy/utils/policy_loader.py` 删除 SmolVLA 的顶层 import、
  loader 分支和专用校验；
- 从部署配置的 `policy_type` 候选中移除 SmolVLA；
- 将 `tools/open_loop_smolvla_eval.py`、
  `tools/open_loop_smolvla_viewer.py` 和对应文档改名为后端无关名称；
- 在重构后的 open-loop 工具中删除 SmolVLA 选项，保留 ACT、DP、
  LingBot-v1、LingBot-v2，并为 OpenPI client 预留统一 adapter；
- 用 lazy import 隔离各模型环境，避免 open-loop 工具导入一个 backend
  时强制安装其他 backend 的依赖。

| 来源提交 | 处置 | 中文说明 |
| --- | --- | --- |
| `4ecf0fe` SmolVLA support | `丢弃` | 最终不再训练或部署 SmolVLA。不得把 SmolVLA loader、配置或依赖带入最终交付。通用 open-loop 能力由后续 open-loop 提交单独保留。 |
| `6a98602` deploy YAML | `丢弃` | 旧示例值和 `.bak` 文件不再权威，保留当前 YAML。 |
| `a1e3b44` open-loop tool | `拆分迁移` | 保留并改造成后端无关的统一 open-loop 工具，最终支持实际交付的 ACT、DP、LingBot-v1、LingBot-v2 和 OpenPI。删除 SmolVLA 专用加载与依赖，保留主线绝对帧索引、raw-row 和 LingBot-v2 行为。 |
| `559136f` EEF/camera YAML | `手工重放` | 仅把仍有效的 EEF 类型和 camera key 默认值移植到当前 schema；删除硬编码本机路径。 |
| `a81e68b` GR00T open loop | `丢弃` | 最终训练和交付不使用 GR00T；删除其 loader、wrapper、action head 和 open-loop 分支。 |
| `06e90bcc` deploy YAML | `丢弃` | classic 后续示例配置修改，当前官方配置继续作为权威版本。 |
| `577ee50` open-loop timeline | `手工重放` | 先与主线 `cd0a527`、`234c70c` 比较，仅重放仍缺失的 viewer 行为。 |
| `ae856c3` gripper latch/diagnostics | `拆分迁移` | diagnostics 已完全一致，不迁移。只移植缺失的 latch 配置和语义，并增加状态转换测试；不为保持历史补一个无内容的 `utils/__init__.py`。 |
| `2993c31` claw command range | `手工重放` | 在当前 `KuavoBaseRosEnv.py` 中恢复夹爪命令边界，覆盖左右夹爪，并确保不破坏 LingBot-v2 绝对动作链。 |
| `3a2612e` action postprocessing/viewer | `拆分迁移` | 将 `action_postprocessing.py` 作为独立共享工具加入；viewer 集成基于当前文件手工实现，不覆盖整个 viewer。 |

B 阶段验收：

- ROS mock 覆盖 ACT、DP、LingBot-v1、LingBot-v2、OpenPI 和 client policy 的初始化。
- 覆盖 timeout、reset、断连、错误 action shape。
- 覆盖 latch 开关状态和左右夹爪命令边界。
- 每条动作路径明确声明 absolute、delta 或 EEF。

### C. ACT/DP loader、训练配置和辅助策略

| 来源提交 | 处置 | 中文说明 |
| --- | --- | --- |
| `8d30d3b` deploy configs | `拆分迁移` | 新增 `dp_r1.yaml`、`dp_r2.yaml`；ACT 配置的两行修改与主线本地 checkpoint 配置加载逻辑手工协调。 |
| `795270e` config reading | `拆分迁移` | `config_loading.py` 已完全一致，不重复迁移。只移植 wrapper/viewer 中主线 `1702d2a` 尚未覆盖的调用点，并测试嵌套在 run 目录中的 checkpoint。 |
| `c90015b` diffusion compile | `手工重放` | 将 compile helper 和配置键重新叠加到主线 trainer，保留主线 LingBot-v2 改动；测试启用、关闭和不支持时回退。 |
| `93c43af` auxiliary EEF diffusion | `丢弃` | 最终训练和交付不使用 EEF diffusion；不迁移其 config、model、dataset、policy、trainer 或 loader 注册。 |
| `82782f2` accelerate multi-GPU | `手工重放` | 在当前 trainer 上移植启动和错误处理，不替换整个文件；进行单进程和双 rank dry-run。 |
| `6dfe942` Task2 H100 profile | `直接迁移` | 在 compile 迁移后加入 H100 YAML 和文档，并验证引用的配置键和路径。 |
| `8c8a5e2` final checkpoint | `手工重放` | 确保 `epochlast` 只保存一次完整可部署 policy 和 processors，同时保留官方 trainer 逻辑。 |
| `40450b5` unused DDP parameters | `直接迁移` | 在 `6dfe942` 后迁移；确认设置与当前 Accelerator/DDP 对接，不能用它掩盖真正未使用的模块。 |

C 阶段验收：

- ACT/DP 配置可以正常组合。
- checkpoint 搜索兼容 run 目录和直接模型目录。
- 最小训练、保存、加载闭环通过。
- compile 开关关闭时不改变基线行为。

### D. Classic Docker 和运行时

| 来源提交 | 处置 | 中文说明 |
| --- | --- | --- |
| `5c3d4ec` Docker fix | `拆分迁移` | 在当前 Dockerfile 上重建有效依赖和运行时调整；丢弃 `Dockerfile.bak`。 |
| `0c8222a` Docker/ignore hygiene | `拆分迁移` | 只合并缺失的 outputs/artifacts 排除项；不得排除当前构建必需的源码和子模块。 |
| `1d1c9eb` num2words | `手工重放` | 只有 import 审计证明 classic runtime 需要时才加入，并在 backend 专用环境固定版本。 |
| `085adde` Docker context exclusions | `拆分迁移` | 合并到单一 hygiene 提交，不保留独立历史提交。 |
| `f0236e4` outputs exclusion fix | `拆分迁移` | 同上，同时核对拼写和 BuildKit context 行为。 |
| `22f7e3d` bundled LeRobot | `拆分迁移` | 加入 `docker/build_classic.sh`；Dockerfile 适配固定的 `third_party/lerobot@56b43cc8`，不得携带 `.bak`、权重或打包环境。 |

D 阶段验收：

- 构建上下文不包含 credentials、outputs、artifacts、checkpoint。
- classic 镜像实际导入固定的 LeRobot 源码。
- 镜像内通过 import smoke test 和一次离线 loader 测试。

### E. 云端训练脚本

| 来源提交 | 处置 | 中文说明 |
| --- | --- | --- |
| `ccbd3c8` A100 pipeline | `直接迁移` | 在训练配置收敛后加入 requirements、脚本和文档；把机器专用默认路径改成显式参数或环境变量。 |
| `37aa243` Task2 downloads | `直接迁移` | 紧跟 `ccbd3c8`；核验下载 ID 和 checksum，但下载产物不得进入 Git。 |

E 阶段验收：

- `bash -n` 和 dry-run 通过。
- 缺少 secret 时快速、安全失败。
- 所有输入输出目录均显式，不默认写入源码目录。

### F. 数据集编辑、QC 和北京数据导出

| 来源提交 | 处置 | 中文说明 |
| --- | --- | --- |
| `760c9c4` episode editor | `直接迁移` | 作为后续 editor 提交的基础版本加入。 |
| `e78a145` editor v10 | `直接迁移` | 在 `760c9c4` 后迁移；初期保留原文件名以维持溯源，同时记录文件名和内部版本不一致。 |
| `20f0c3f` multithread conversion | `拆分迁移` | 将转换器修改手工重放到官方 `CvtRosbag2Lerobot.py`，保留官方 chunk/timestamp 修复，丢弃 `.bak`。 |
| `bde4812` geometry annotations | `丢弃` | 这些工具服务于本轮已取消的 EEF diffusion/几何监督链路，与最终训练交付无关。 |
| `9cae901` QC/rebuild tools | `直接迁移` | 在 editor 基础提交后迁移，执行 import、`--help` 和 header alignment fixture 测试。 |
| `9b13f56` Beijing audit/export | `直接迁移` | 在 `9cae901` 后迁移；审计命令默认只读，导出命令必须显式指定输出目录，并生成确定性 metadata。 |

F 阶段验收：

- fixture 转换保持时间戳、状态和动作对齐。
- audit 工具默认只读。
- exporter 不覆盖源数据，并要求显式输出目录。

### G. LingBot-v1 云端流水线

| 来源提交 | 处置 | 中文说明 |
| --- | --- | --- |
| `128f896` full Task1 LingBot pipeline | `拆分迁移` | 新增 config、docs、requirements、脚本和 checkpoint export 工具；两个已完全一致的 compat 文件不重复迁移。trainer 和 norm-stat 修改基于 `305f78a` 手工重放，保留后续 LingBot-v2 兼容和保护逻辑。声明可复现前，先固定 LingBot-v1 `4eb34b7` 及其两个嵌套 gitlink。 |

G 阶段验收：

- bundle 脚本拒绝未固定或缺失的 LingBot 源码。
- bundle 不收集凭据。
- 配置组合和 checkpoint export dry-run 通过。

### H. 明确排除的文档

| 来源提交 | 处置 | 中文说明 |
| --- | --- | --- |
| `f675e82` interview deep dives | `丢弃` | 已确认不进入 unified 仓库。 |

## 四、重点冲突文件的所有权

| 文件或目录 | 权威基础 | 冲突处理方案 |
| --- | --- | --- |
| `Dockerfile`、`.dockerignore`、`.gitignore` | `305f78a` 当前构建与子模块要求 | 手工加入 classic backend；不接收 `.bak`，不扩大 artifact context。 |
| `configs/deploy/kuavo_env.yaml` | 官方 schema 和 LingBot-v2 动作契约 | 逐键迁移，不整文件选择。 |
| `kuavo_deploy/config.py` | 官方 policy/client 类型和校验 | 只添加经过校验的 classic 字段，保留 client 与 LingBot-v2。 |
| `real_single_test.py`、`sim_auto_test.py` | 官方 timeout/reset/client/diagnostics 流程 | action/latch 只在 postprocessor 后、物理执行前接入一次。 |
| `kuavo_deploy/utils/policy_loader.py` | 官方 loader | 最终只保留 ACT、DP、LingBot-v1、LingBot-v2、OpenPI/client 所需入口；移除 SmolVLA、GR00T 和 EEF diffusion。可选依赖不得在 import 阶段使其他 backend 失败。 |
| `KuavoBaseRosEnv.py` | 官方物理命令语义 | 只移植 claw bounds；不得恢复 LingBot-v2 二次 delta-to-absolute。 |
| `train_policy.py`、`train_policy_with_accelerate.py` | 官方 trainer 和 LingBot-v2 launch 行为 | 手工叠加 compile、DDP、final-save，并各自测试。 |
| LingBot-v1 trainer/norm 工具 | `305f78a` 文件版本 | 只重放 cloud pipeline 所需增量，保留官方兼容加固。 |
| open-loop viewer/eval | 官方 absolute-frame、raw-row 和 LingBot-v2 行为 | 只加入独立面板和诊断能力，不整文件覆盖。 |

## 五、来源保护前置条件

已完成：

1. unified 已配置个人 remote `personal`：
   fetch 使用 `https://github.com/Larryi/kuavo-unified.git`，push 使用 SSH。
2. OpenPI 已配置个人 remote `fork`，并已把
   `kuavo-lerobot-v3@7f8010a` 发布到 `Larryi/openpi`。
3. 已导入并固定：
   - LeRobot `56b43cc8`；
   - LingBot-v1 `4eb34b7`；
   - LingBot-v2 `d34898d`（基于官方 `2838c18`，包含稀疏目标槽位补丁和 compact/right-arm/bimanual 回归测试）；
   - OpenPI `7f8010a`。
4. 已从本地只读 classic 工作树建立
   `source/ship/classic@f675e82` 跟踪引用。
5. LingBot-v2 上游从旧基线 `894c40a` 更新到 `2838c18` 后仍未吸收
   `target_start` 修复。该修复已在隔离克隆中重放，新增右臂稀疏槽位
   回归测试，并提交为
   `Larryi/lingbot-vla-v2:codex/kuavo-target-slots@d34898d`。
   测试确认普通单臂保持紧凑映射、双臂保持拼接映射，同时右臂关节写入
   55 维空间的 `7..13`、右夹爪写入槽位 `1`。

## 六、需要审核确认的决策

已确认：

1. 不保留 SmolVLA 训练、部署、loader 或依赖；只保留并统一通用
   open-loop 测试能力。
2. 丢弃 GR00T（`a81e68b`）。
3. 丢弃 EEF diffusion 及其几何监督工具（`93c43af`、`bde4812`）。
4. 不迁移 interview 文档（`f675e82`）。
5. unified 使用 `Larryi/kuavo-unified`，OpenPI 使用 `Larryi/openpi`。
6. 允许导入固定版本子模块；LingBot-v2 应先在最新上游
   `2838c18` 上形成可追溯的 `target_start` 补丁提交。该项已完成，
   固定提交为 `d34898d`。

上述范围决策已经审核确认。用户随后已明确授权按本计划继续创建逐提交
迁移；仍不得向官方仓库推送，也不得把已明确丢弃的功能重新带入。

## 七、执行状态（2026-07-25）

- 来源保护与子模块固定已完成，并通过 fresh-clone 的 9 项基础测试。
- B 阶段代码迁移已完成：删除 SmolVLA 交付入口，保留后端无关
  open-loop；加入因果限速/边界平滑、跨调用夹爪 latch、Leju 左右夹爪
  0–80 边界和 viewer 原始/处理后诊断。GR00T、EEF diffusion 均未引入。
- C 阶段代码迁移已完成：加入 R1/R2 DP 配置、本地数据集离线保护、
  默认关闭且可回退的 `torch.compile`、Accelerate 多 rank 加固、
  Task 2 H100 配置和单次 `epochlast` 交付 checkpoint。
- 已通过 28 项模块/回归测试、YAML/Hydra 组合检查、viewer 导入检查、
  单进程 smoke 和双 rank CPU 同步 smoke。
- Gate C 仍为“部分通过”：当前执行环境无可用 CUDA，尚未执行真实
  DP 最小训练→保存→重载闭环。该项必须在 GPU 环境补测后才能把训练
  链路标记为交付就绪。
- D（Docker）、E（云端训练脚本）、F（数据编辑/QC/北京导出）和
  G（LingBot-v1 云端流水线）尚未开始。

因此当前分支适合作为可复现的继续集成基线，但还不是最终训练/部署
交付版本。
