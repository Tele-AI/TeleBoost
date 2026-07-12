<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/logo_teleboost.jpeg">
    <img alt="TeleBoost" src="docs/figures/logo_teleboost.jpeg" width="55%">
  </picture>
</p>
<h3 align="center">
统一的扩散模型 post-training 框架
</h3>

<p align="center">
  <a href="https://tele-ai.github.io/TeleBoost/"><img alt="Project page" src="https://img.shields.io/badge/Project_page-tele--ai.github.io-4C1?labelColor=555555"></a>
  <a href="https://arxiv.org/abs/2602.07595"><img alt="TeleBoost arXiv" src="https://img.shields.io/badge/TeleBoost-arXiv%202602.07595-B31B1B?labelColor=555555"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-2196F3?labelColor=555555"></a>
  <a href="https://github.com/Tele-AI/TeleBoost/actions/workflows/cpu-tests.yml"><img alt="CPU tests" src="https://github.com/Tele-AI/TeleBoost/actions/workflows/cpu-tests.yml/badge.svg"></a>
  <a href="https://arxiv.org/abs/2511.18919"><img alt="BGPO CVPR 2026" src="https://img.shields.io/badge/BGPO-CVPR%202026-1A73E8?labelColor=555555"></a>
  <a href="https://arxiv.org/abs/2511.18719"><img alt="VIPO CVPR 2026" src="https://img.shields.io/badge/VIPO-CVPR%202026-1A73E8?labelColor=555555"></a>
</p>

<p align="center"><a href="README.md">English</a> | 中文</p>

TeleBoost 是一个**统一的扩散模型 post-training 框架**，支持
**DPO** 和 **GRPO**。已在 TeleAI 内部用于扩散模型对齐训练。

* 🎛️ **多范式后训练** —— 支持 DPO + GRPO
* 🔥 **显存高效 DPO** —— Wan 14B 上峰值显存降 ~40%、上下文扩 ~15×
* 🆕 **六种 GRPO 算法** —— DanceGRPO、Flow-GRPO、GRPO-Guard、TempFlow-GRPO、**BGPO**、**VIPO**
* 🧩 **Co-located + MPS 多奖励** —— N 路奖励同卡执行，墙钟 ≈ max(model)
* 🎬 **即开即用的序列并行** —— Ulysses SP，面向长视频训练
* 🚀 **Day-0 BGPO + VIPO（CVPR 2026）**

<p align="center">
  <img src="docs/figures/fig_memory_vs_layers.png" alt="Wan 14B DPO 峰值显存：Gradient Decoupled DPO 在相同负载下削减约 40% 显存，支持约 15× 于标准方案的上下文长度。" width="720"/>
</p>
<p align="center"><sub><i>
Wan 14B DPO 在 32× 80GB-Hopper GPUs 上的峰值显存 —— Decoupled 方案在相同负载下
削减 <b>~40%</b> 峰值显存，并支持约 <b>~15×</b> 长上下文。
详见上方 badge 中的项目主页。
</i></sub></p>

## 方法

<div align="center">

| 方法 | 状态 | 用途 | 路径 |
|:-----|:----:|:-----|:-----|
| **DPO** | ✅ 已就绪 | 偏好对齐 | [`recipes/wan_dpo_teletron/`](recipes/wan_dpo_teletron/) |
| **GRPO** | ✅ 已就绪 | 基于奖励的优化 | [`recipes/wan_grpo_fsdp/`](recipes/wan_grpo_fsdp/) |
| **TempFlow-GRPO** | ✅ 已就绪 | 噪声感知加权 + 轨迹分支([arXiv 2508.04324](https://arxiv.org/abs/2508.04324)) | [`recipes/wan_tempflow_fsdp/`](recipes/wan_tempflow_fsdp/) |
| **GRPO-Guard** | ✅ 已就绪 | 抑制隐式过优化的正则化裁剪([arXiv 2510.22319](https://arxiv.org/abs/2510.22319));可组合的算法能力,非独立 recipe | [`teleboost/algorithms/grpo_guard.py`](teleboost/algorithms/grpo_guard.py) |
| **BGPO** | ✅ 已就绪 | 贝叶斯先验组优化(CVPR 2026) | [`recipes/wan_bgpo_fsdp/`](recipes/wan_bgpo_fsdp/) |
| **VIPO** | ✅ 已就绪 | 像素加权稠密 advantage(CVPR 2026) | [`recipes/wan_vipo_fsdp/`](recipes/wan_vipo_fsdp/) |
| DPO 的 FSDP 后端 | 🚧 Roadmap | 不依赖 DeepSpeed-ZeRO 的显存分片 | — |

</div>

DPO recipe 提供了对独立 megatron 参考实现的精度对齐锚点。
详见 [`recipes/wan_dpo_teletron/README.md`](recipes/wan_dpo_teletron/README.md)。

## 快速开始

选择匹配你后训练需求的 recipe。顶层 README 用于了解能力全貌；
具体命令、环境变量和数据要求请进入对应 recipe 文档。

* **GRPO** —— 见 [`INSTALL.md`](INSTALL.md)。设置
  `TRAIN_FILE` / `TEST_FILE` / `WAN_MODEL_PATH` / `REWARD_MODEL_PATH`
  后执行 `bash recipes/wan_grpo_fsdp/run.sh`（冒烟跑可用
  `TEST_FILE=$TRAIN_FILE`）。
* **DPO** —— 见
  [`recipes/wan_dpo_teletron/README.md`](recipes/wan_dpo_teletron/README.md)。
  构建 Dockerfile 并导出 `MEGATRON_LM_DIR` 后执行
  `bash recipes/wan_dpo_teletron/run.sh`。

### 文档导航

| 需求 | 从这里开始 |
|:-----|:-----------|
| 了解支持的算法和系统能力 | 本 README 与 [`SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md) |
| 安装并运行 GRPO 训练 | [`INSTALL.md`](INSTALL.md) |
| program 清单与各 recipe 的运行方式 | [`recipes/README.md`](recipes/README.md) |
| 理解 DPO 模式、精度对齐和排障 | [`recipes/wan_dpo_teletron/README.md`](recipes/wan_dpo_teletron/README.md) |

---

## 🚀 TeleAI 新进展

本仓库包含 TeleAI 的四项贡献：**VIPO** + **BGPO**（day-0 GRPO
论文）、**co-located reward + MPS**（GRPO 系统）、**Gradient
Decoupled DPO**（DPO 系统）。

### VIPO —— Visual Preference Policy Optimization &nbsp;·&nbsp; *GRPO* &nbsp;·&nbsp; CVPR 2026 &nbsp;·&nbsp; [arXiv 2511.18719](https://arxiv.org/abs/2511.18719)

把 GRPO 的标量反馈提升为**结构化、像素级**的优势 —— 通过感知结构化
模块产出空间感知的优势图。详见 [arXiv:2511.18719](https://arxiv.org/abs/2511.18719)。

<p align="center">
  <img src="docs/figures/vipo_method.png" alt="VIPO 流程图：标准 GRPO 使用标量 advantage；VIPO 改为按像素 / 区域分配的结构化 advantage。" width="780"/>
</p>
<p align="center"><sub><i>
<b>上图</b> —— 标准 GRPO：奖励模型输出在策略更新前压缩为标量
advantage。<b>下图</b> —— VIPO：偏好信号被分配为结构化的 advantage
图，将优化压力导向感知上重要的区域。
</i></sub></p>

### BGPO —— Bayesian-Prior Group Optimization &nbsp;·&nbsp; *GRPO* &nbsp;·&nbsp; CVPR 2026 &nbsp;·&nbsp; [arXiv 2511.18919](https://arxiv.org/abs/2511.18919)

基于**贝叶斯先验**的两级优化：组间**信任分配**（RAS）+ 组内
**先验锚定重标定**（CRT）。详见 [arXiv:2511.18919](https://arxiv.org/abs/2511.18919)。

<p align="center">
  <img src="docs/figures/bgpo_method.png" alt="BGPO 流程图：RAS（左）从组内奖励与贝叶斯先验得出逐样本信任权重；CRT（右）以先验为基线对奖励重标定后再驱动下一轮 GRPO。" width="780"/>
</p>
<p align="center"><sub><i>
BGPO 基于贝叶斯先验在两个层级上运作。<b>左（RAS）</b>：组内奖励 +
先验得到逐样本可靠性权重 → 可靠性感知损失 <code>ℒ_RAS</code>。
<b>右（CRT）</b>：以先验为基线对奖励重标定 → 重标定后的信号驱动
下一轮 GRPO 损失 <code>ℒ_CTR</code>。
</i></sub></p>

### Co-located reward + MPS 并行多奖励 &nbsp;·&nbsp; *GRPO 系统*

**Co-located reward**（worker 与 actor 共用 GPU）**+ MPS 并行多
奖励**（N 路奖励同卡通过 CUDA MPS 并发）。消除独立 reward rank
的空转，联合墙钟 ≈ max(model) 而非求和。joint 模式默认开启。


<p align="center">
  <img src="docs/figures/colocate_mps.png" alt="两项互补的吞吐优化。左：co-located reward —— reward worker 与 actor 共用 GPU，消除独立 reward rank 在 rollout / 训练切换时的空转。右：CUDA MPS —— N 路奖励模型在同一张卡上并发计算，墙钟 ≈ max(model) 而非求和。" width="780"/>
</p>
<p align="center"><sub><i>
<b>左</b>：reward worker 与 actor 共用 GPU，消除独立 reward rank
在 rollout / 训练切换时的空转。<b>右</b>：CUDA MPS 让 N 个奖励模型
同卡并发执行，整体耗时 ≈ 最慢那个模型。
</i></sub></p>

### Gradient Decoupled DPO &nbsp;·&nbsp; *DPO 系统*

逐分支反向 + 即时 **reduce-scatter** —— 在下一次反向开始前释放
本路全形状梯度。与单次反向**数学等价**；在 Wan 14B DPO @ 32× 80GB-Hopper GPUs
上**峰值显存降 ~40%**、**上下文扩 ~15×**。

<p align="center">
  <img src="docs/figures/fig_dpo_mechanism.png" alt="标准 DPO 与 Gradient Decoupled DPO 的反向时间轴对比：标准方案同时持有 chosen / rejected 两路全形状梯度；Decoupled 方案在每路反向完成后立即将其 reduce-scatter 到本 rank 的 1/N 分片。" width="780"/>
</p>
<p align="center"><sub><i>
逐分支反向 + 即时 reduce-scatter。Decoupled DPO 在每路分支反向完成
后立即释放全形状张量，从而显著降低峰值。（结果图见 README 顶部。）
</i></sub></p>

---

## 仓库结构

```
teleboost/          唯一生产 Python 包
  programs/         composition root:ProgramSpec 绑定 模型家族×算法×引擎×运行策略
  engines/          分布式执行引擎(fsdp、teletron/Megatron)
  training/         中立训练骨架(core/)+ family 训练适配(families/)
  algorithms/       算法数学(grpo、bgpo、vipo、tempflow、grpo_guard、…)
  models/           Wan 模型与 family 语义(attention、采样、转换)
  reward/           奖励契约、执行与 providers
  datasets/         数据集、transforms 与 Wan 数据预处理
  cli/              安装后命令入口
  artifacts/        checkpoint 产物转换
  config/ patches/  配置加载与钉版上游补丁
recipes/            声明式配置 + 启动脚本;teleboost 永不 import 它
third_party/        vendored 上游源码(独立 license,不进发布物)
tools/              安装 / 发布 / 冒烟 / 诊断脚本(不进 wheel)
tests/              core / training / heavy(wan)三档 profile
docs/               figures 与架构文档

Dockerfile / makefile / pyproject.toml / requirements.txt   构建 + 依赖
LICENSE / NOTICE / CITATION.cff                              上游归因
.github/                                                     CI + CODEOWNERS
```

各 program 的运行入口是 `recipes/<program>/run.sh`;program 清单见
[`recipes/README.md`](recipes/README.md),依赖方向与所有权边界见
[`docs/target_architecture.md`](docs/target_architecture.md)。

## 许可证

TeleBoost 以 **Apache 2.0** 开源 —— 见 [`LICENSE`](LICENSE)。

## 致谢

本项目站在以下上游之上构建。完整逐项归因（license 全文 + 再分发
条款）见 [`NOTICE`](NOTICE)。

**RL 训练栈**

* [`volcengine/verl`](https://github.com/volcengine/verl) —— Apache 2.0。字节跳动的 RL 训练框架；TeleBoost 作为 recipe 层构建于其上。
* [`DanceGRPO`](https://github.com/XueZeyue/DanceGRPO) —— Apache 2.0。TeleBoost 的 GRPO 系算法实现的是 DanceGRPO 发表的方法（[arXiv 2505.07818](https://arxiv.org/abs/2505.07818)）；脚手架直接构建于上游 verl。
* [`Tele-AI/TeleTron`](https://github.com/Tele-AI/TeleTron) —— Apache 2.0。TeleAI 的长上下文多模态训练框架；TeleBoost 在其之上加入 Gradient Decoupled DPO。

**生成模型**

* [`Wan-Video/Wan2.1`](https://github.com/Wan-Video/Wan2.1) —— Apache 2.0。阿里的 Wan2.1 / Wan2.2 视频扩散模型，按上游 `LICENSE` 内嵌于 `third_party/wan/`。

**奖励模型**

* [`tgxs002/HPSv2`](https://github.com/tgxs002/HPSv2) —— Apache 2.0。Human Preference Score v2。
* [`alibaba-pai/VideoCLIP-XL`](https://huggingface.co/alibaba-pai/VideoCLIP-XL) —— CC-BY-NC-SA-4.0（禁商用）。阿里的视频-文本对齐模型。本仓库不再分发其代码；`videoclip` 奖励会加载用户自行放置在 `third_party/VideoCLIP_XL/` 下的副本。
* [`Hritikbansal/videophy`](https://github.com/Hritikbansal/videophy) —— MIT。UCLA 的视频物理合理性模型。
* [`LAION-AI/aesthetic-predictor`](https://github.com/LAION-AI/aesthetic-predictor) —— MIT。LAION 的 CLIP + 线性头美学预测器。
* [`princeton-vl/RAFT`](https://github.com/princeton-vl/RAFT) —— BSD-3-Clause。Princeton 的光流模型，用作时序一致性奖励。
* [`TencentARC/VideoAlign`](https://github.com/TencentARC/VideoAlign) —— Apache 2.0。参考其奖励模型设计；本仓库未内嵌(需要的话从上游拉)。

## 引用

```bibtex
@article{teleboost2026,
  title  = {TeleBoost: A Systematic Alignment Framework for High-Fidelity,
            Controllable, and Robust Video Generation},
  author = {Liang, Yuanzhi and Wu, Xuan'er and Liu, Yirui and Fang, Yijie and
            Fan, Yizhen and Hao, Ke and Li, Rui and Liu, Ruiying and Ni, Ziqi and
            Yu, Peng and Wang, Yanbo and Huang, Haibin and Weng, Qizhen and
            Zhang, Chi and Li, Xuelong},
  year   = {2026},
}
```

各算法的具体引用：

* **GRPO 类算法**（DanceGRPO、Flow-GRPO、GRPO-Guard、**BGPO**、**VIPO**）—— 见 [`CITATION.cff`](CITATION.cff)。
* **Gradient Decoupled DPO** —— 引用上方 TeleBoost 论文。
