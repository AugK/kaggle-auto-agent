# kaggle-auto-agent

**教算法工程师/转行人用 AI Agent 打竞赛、攒作品集的公开方法论 + 可复用模板。**

> English version: [README_EN.md](README_EN.md)

在一场 3500+ 队的表格赛里，我们用"一个人 + AI agent"的打法打进了前 10%。
这个仓库是那套打法的公开版：**判断层全公开、方法层免费教、系统层另议**。

## 这个仓库是什么

一个"用 agent 打表格赛"的最小可运行框架：

```
agent/
├── knowledge_base.py      # 知识库：六条普适原则 + 带适用条件的战术表
├── planner.py             # 规划器：阶段化实验计划（基线→特征→模型→融合，每步限时）
├── executor.py            # 执行器：协议化实验 + OOF 缓存 + 折内门控
├── evaluator.py           # 评估器：delta → 判决（adopt/candidate/undecidable/reject）
├── decision_engine.py     # 决策引擎：何时深挖、何时换向、何时转融合
├── ensemble_optimizer.py  # 融合优化（设计占位，见文件 docstring）
└── agent.py               # 编排循环（设计占位，见文件 docstring）
core/
├── common.py              # 数据配置、冻结折、统一 CV、实验日志
├── features.py            # 折内 TE / 分箱 / 交互（防泄漏纪律在这里）
└── dataset.py             # 特征集组装：raw / inter / te / combo
configs/demo.example.json  # 数据配置模板（复制为 demo.json 后填你的路径）
docs/
├── tutorial.md             # 五步法完整教程（喂 AI 的工作流输入之一）
└── principles.md           # 六条原则详解（喂 AI 的工作流输入之二）
```

## 快速开始（3 分钟）

```bash
pip install -r requirements.txt
cp configs/demo.example.json configs/demo.json
# 编辑 configs/demo.json：填入你的数据路径与列名
# 3. 跑一个基线
python -c "
from core.common import load_config
from agent.executor import ExperimentExecutor
ex = ExperimentExecutor(load_config('demo'), n_jobs=8)
print(ex.protocol_baseline('raw'))"
```

## 让 AI 帮你从 0 搭（推荐用法）

本项目的设计目标不是"下载即用"，而是**在 AI 辅助下自己搭一遍**——搭的过程才是
作品集的价值。打开 [PROMPT.md](PROMPT.md)，把它连同 `docs/` 两个文档喂给任意
AI 编程助手（Claude Code / Cursor / 豆包电脑版都行），按五步法逐模块生成，
每个模块生成后和本仓库的参照实现对照。

## 感知层（可选但强烈推荐）：tools/kaggle_mcp.py

agent 的"每天扫榜、监控社区"不是比喻——它需要一个能读 Kaggle 的通道。
`tools/kaggle_mcp.py` 是 Kaggle 官方 MCP 服务的轻量 CLI 封装（106 行纯标准库，
零第三方依赖），管"读"：比赛元数据、Rules/Overview 全文、讨论区、榜单、
提交状态、公开 notebook 检索。

**前置**：Kaggle 账号 + access token（`~/.kaggle/access_token`，与 kaggle CLI
共用同一个 token）。

三条日常用法（真实工作流）：

```bash
# 1) 新比赛侦察三连：元数据一条全出 → 规则/评测页全文 → 讨论区话题
python tools/kaggle_mcp.py get_competition '{"request": {"competitionName": "titanic"}}'
python tools/kaggle_mcp.py list_competition_pages '{"request": {"competitionName": "titanic"}}'
python tools/kaggle_mcp.py list_competition_topics '{"request": {"competitionName": "titanic"}}'

# 2) 每日开工：自己的排名 + 社区新方案
python tools/kaggle_mcp.py get_competition '{"request": {"competitionName": "titanic"}}'
python tools/kaggle_mcp.py list_competition_topics '{"request": {"competitionName": "titanic"}}'

# 3) 提交后轮询出分
python tools/kaggle_mcp.py get_competition_submission '{"request": {"ref": 123456}}'
```

**分工与坑**（实战判例）：MCP 只管"读"。**提交走 kaggle CLI**
（`kaggle competitions submit`）；数据集更新、`save_notebook` 有已知坑（会静默
丢比赛数据源），一律走 CLI 不走 MCP。参数必须嵌套 `request`；响应是 SSE 分块。

## 内容分层声明

| 层 | 内容 | 获取方式 |
| --- | --- | --- |
| 决策层 | 选赛逻辑、设计取舍、推翻 agent 方案的经过、错误复盘 | 全公开（持续发布） |
| 方法层 | 模块思路、教程、可复用模板（本仓库） | 免费使用，MIT 协议 |
| 系统层 | 完整集成流水线（编排循环+融合优化实现）、持续更新、答疑 | 暂未开放 |

**仓库是静态快照。** 融合器与编排循环只有设计说明，不含实现——完整集成
流水线与随赛季进化的战术库暂未开放。

## 测试

```bash
pip install pytest
python -m pytest tests/test_smoke.py -v   # 17 个冒烟测试，~3 分钟
```

改动代码后必须跑一遍：17 个测试覆盖每个公开函数与每条特征/执行路径。

## License

MIT License，完整文本见 [LICENSE](LICENSE)。

简言之：可自由使用、修改、分发（含商业用途），唯一要求是保留版权与许可声明。
建议在衍生内容中注明出处，但这属于社区礼仪而非法律义务。
