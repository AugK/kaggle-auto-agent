# kaggle-auto-agent

**Public methodology + reusable templates for competing with AI agents — build a portfolio-worthy Kaggle project as a solo, then write it into your resume.**

In a 3,500+ team tabular competition, a solo participant driven by AI agents
finished top 10%. This repo is the public version of that playbook:
**judgment layer fully public, methodology layer free, system layer not yet available.**

> 中文文档：见 [README.md](README.md)（其余 docs/ 为中文）。

## What this is

A minimal, runnable framework for "tabular competition with an agent":

```
agent/
├── knowledge_base.py      # KB: six universal principles + tactics with conditions/risks
├── planner.py             # Planner: staged experiment plan (baseline→features→models→blending, time-boxed)
├── executor.py            # Executor: protocolized runs + OOF cache + fold gate
├── evaluator.py           # Evaluator: delta → verdict (adopt/candidate/undecidable/reject)
├── decision_engine.py     # Decision engine: when to deepen, pivot, or start blending
├── ensemble_optimizer.py  # Ensemble blending (design stub — see module docstring)
└── agent.py               # Orchestration loop (design stub — see module docstring)
core/
├── common.py              # Data config, frozen folds, unified CV, experiment log
├── features.py            # Fold-safe TE / binning / interactions (leak discipline lives here)
└── dataset.py             # Feature set assembly: raw / inter / te / combo
configs/demo.example.json  # Data config template (copy to demo.json, fill in your paths)
docs/
├── tutorial.md            # Full five-step tutorial (Chinese)
└── principles.md          # Six principles explained (Chinese)
tools/
└── kaggle_mcp.py          # Perception layer: CLI wrapper for the Kaggle MCP server
```

## Quick start (3 minutes)

```bash
pip install -r requirements.txt
cp configs/demo.example.json configs/demo.json
# Edit configs/demo.json: your competition data paths and column names
python -c "
from core.common import load_config
from agent.executor import ExperimentExecutor
ex = ExperimentExecutor(load_config('demo'), n_jobs=8)
print(ex.protocol_baseline('raw'))"
```

## Let an AI build it with you (recommended)

The point of this repo is not "download and run" — it is **building it yourself
with an AI assistant**. That build is what makes it portfolio-worthy.
Open [PROMPT.md](PROMPT.md), feed it together with the two docs under `docs/`
to any coding assistant (Claude Code / Cursor / etc.), and go module by module,
comparing each result against the reference implementation in this repo.

## Perception layer (optional but recommended): tools/kaggle_mcp.py

The agent's daily "scan the community" routine needs a way to read Kaggle.
`tools/kaggle_mcp.py` is a lightweight CLI wrapper around the official Kaggle
MCP server (106 lines, stdlib only, zero third-party deps). It covers the
"read" side: competition metadata, Rules/Overview pages, discussion topics,
leaderboard, submission status, public notebook search.

**Prerequisite**: a Kaggle account + access token (`~/.kaggle/access_token`,
same token the kaggle CLI uses).

Three daily patterns (real workflow):

```bash
# 1) New competition recon trio
python tools/kaggle_mcp.py get_competition '{"request": {"competitionName": "titanic"}}'
python tools/kaggle_mcp.py list_competition_pages '{"request": {"competitionName": "titanic"}}'
python tools/kaggle_mcp.py list_competition_topics '{"request": {"competitionName": "titanic"}}'

# 2) Daily standup: your rank + fresh community solutions
python tools/kaggle_mcp.py get_competition '{"request": {"competitionName": "titanic"}}'
python tools/kaggle_mcp.py list_competition_topics '{"request": {"competitionName": "titanic"}}'

# 3) Poll a submission for its score
python tools/kaggle_mcp.py get_competition_submission '{"request": {"ref": 123456}}'
```

**Division of labor and gotchas** (field-tested): the MCP wrapper reads only.
**Submissions go through the kaggle CLI** (`kaggle competitions submit`).
Dataset updates and `save_notebook` have known pitfalls (it silently drops the
competition data source) — always use the CLI for those. Parameters must be
nested under `request`; responses are SSE-chunked.

## Content layers

| Layer | Contents | How to get it |
| --- | --- | --- |
| Judgment | competition selection logic, design trade-offs, overruling the agent, post-mortems | fully public (ongoing on social media) |
| Method | module design, tutorials, reusable templates (this repo) | free to use, MIT |
| System | full integrated pipeline (orchestration + blender implementation), per-season updates, Q&A support | see author profile |

**This repo is a static snapshot.** The blender and orchestration loop ship as
design stubs without implementations — the full integrated pipeline and a
tactics library that evolves each season are not yet available.

## Tests

```bash
pip install pytest
python -m pytest tests/test_smoke.py -v   # 17 smoke tests, ~3 min
```

Run after any change: 17 tests cover every public function and every
feature/execution path.

## License

MIT License — see [LICENSE](LICENSE) for the full text.

In short: free to use, modify, and distribute (including commercially). The
only requirement is to retain the copyright and license notice. Attribution
in derivative content is appreciated but not legally required.
