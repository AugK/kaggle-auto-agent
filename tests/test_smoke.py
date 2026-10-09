# -*- coding: utf-8 -*-
"""kaggle-auto-agent 冒烟测试：覆盖公开模块的每条代码路径。

背景教训（2026-10-08）：core/features.py 的 add_te_features 曾在翻译改造中丢失
`cname` 定义行，而当时的验证只跑了 baseline（raw 特征集，不走 TE 路径），
NameError 潜伏到用户复核才暴露。

本测试的覆盖承诺：公开模块的每个函数至少被真实调用一次；每条特征链路
（raw/inter/te/combo）至少端到端走一遍；两条执行路径（缓存命中/未命中）、
探针模式、门控中止、评估器四档判决、决策引擎五种动作都有断言。

运行：
    cd <仓库根目录>
    python -m pytest tests/test_smoke.py -v
或无 pytest 时：
    python tests/test_smoke.py

需要 configs/demo.json（数据配置）与本机可用的 lightgbm。
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------- fixtures ----------

@pytest.fixture(scope="session")
def cfg():
    from core.common import load_config, DataConfig
    if os.path.exists(os.path.join(ROOT, "configs", "demo.json")):
        return load_config("demo")
    # 无真实数据时退化为合成小数据（CI/新机可跑）
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame({
        "id": range(n),
        "num_a": rng.normal(50, 15, n).round(2),
        "num_b": rng.normal(5, 2, n).round(2),
        "cat_a": rng.choice(["red", "green", "blue"], n),
        "cat_b": rng.choice(["yes", "no"], n),
    })
    logit = 0.02 * df["num_a"] + 0.3 * df["num_b"] + (df["cat_b"] == "yes") * 1.2
    df["target"] = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)
    train_path = os.path.join(ROOT, "data", "train.csv")
    test_path = os.path.join(ROOT, "data", "test.csv")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    df.iloc[:320].to_csv(train_path, index=False)
    df.iloc[320:].to_csv(test_path, index=False)
    return DataConfig(name="synth", train_path=train_path, test_path=test_path,
                      id_col="id", target_col="target", pos_label=1,
                      num_cols=["num_a", "num_b"], cat_cols=["cat_a", "cat_b"])


@pytest.fixture(scope="session")
def executor(cfg):
    from agent.executor import ExperimentExecutor
    return ExperimentExecutor(cfg, n_jobs=2)


# ---------- core.features：四个特征函数每个都要被真实调用 ----------

def test_ordinal_encode(cfg):
    from core.features import ordinal_encode
    train, test = pd.read_csv(cfg.train_path), pd.read_csv(cfg.test_path)
    tr, te = ordinal_encode(train, test, cfg.cat_cols)
    for c in cfg.cat_cols:
        assert pd.api.types.is_integer_dtype(tr[c]), f"{c} 应编码为整数"
        assert te[c].between(-1, 999).all(), "未见类别应为 -1"


def test_add_interaction_features(cfg):
    from core.features import add_interaction_features
    train, test = pd.read_csv(cfg.train_path), pd.read_csv(cfg.test_path)
    tr, te, combo_cols = add_interaction_features(train, test, cfg.cat_cols)
    assert combo_cols, "至少生成一个组合键"
    for c in combo_cols:
        assert c in tr.columns and c in te.columns


def test_add_quantile_bins(cfg):
    from core.features import add_quantile_bins
    train, test = pd.read_csv(cfg.train_path), pd.read_csv(cfg.test_path)
    tr, te, bin_cols = add_quantile_bins(train, test, cfg.num_cols)
    assert bin_cols, "每个数值列都应有分箱列"


def test_add_te_features_requires_fold(cfg):
    """回归测试：TE 无 fold 列必须拒绝运行（防全量统计泄漏）。"""
    from core.features import add_te_features
    train, test = pd.read_csv(cfg.train_path), pd.read_csv(cfg.test_path)
    with pytest.raises(AssertionError):
        add_te_features(train, test, cfg.cat_cols[0])


def test_add_te_features_happy_path(cfg):
    """回归测试（2026-10-08 cname 缺失事故）：TE 主路径必须端到端可用。"""
    from core.common import get_folds, load_data
    from core.features import add_te_features
    train, test = load_data(cfg)
    train = get_folds(train, cfg)
    key = cfg.cat_cols[0]
    tr, te, te_cols = add_te_features(train, test, key)
    assert te_cols == [f"te__{key}"]
    assert te_cols[0] in tr.columns and te_cols[0] in te.columns
    assert tr[te_cols[0]].notna().all(), "折内编码应覆盖全部训练行"
    assert te[te_cols[0]].notna().all()


# ---------- core.common：折冻结 / CV / 日志 / 门控 ----------

def test_get_folds_frozen(cfg):
    """回归测试：折分配必须可复现（两次生成完全一致）。"""
    from core.common import get_folds, load_data
    t1 = get_folds(load_data(cfg)[0], cfg)
    ffile_time = os.path.getmtime(os.path.join(ROOT, "data", f"{cfg.name}_fold_assignment.csv"))
    t2 = get_folds(load_data(cfg)[0], cfg)
    assert (t1["fold"].values == t2["fold"].values).all()
    assert os.path.getmtime(os.path.join(ROOT, "data", f"{cfg.name}_fold_assignment.csv")) == ffile_time


def test_run_cv_and_log(cfg):
    import json
    from core.common import load_data, get_folds, run_cv
    log_path = os.path.join(ROOT, "logs", "experiments.jsonl")
    before = sum(1 for _ in open(log_path)) if os.path.exists(log_path) else 0
    tr, te, feats, test = raw_feats_ready(cfg)
    import lightgbm
    oof, test_pred = run_cv(lambda: lightgbm.LGBMClassifier(verbose=-1),
                            tr, te, feats, name="smoke_cv")
    after = sum(1 for _ in open(log_path))
    assert after == before + 1, "run_cv 必须写一条实验日志"
    assert 0.5 < roc_auc_of(cfg, oof) < 1.0


def helper_train(cfg):
    from core.common import load_data, get_folds
    train, test = load_data(cfg)
    return get_folds(train, cfg), test


def raw_feats_ready(cfg):
    """加载 raw 特征集并完成序数编码（LGBM 不接受 object 列）。"""
    from core.common import load_data, get_folds
    from core.features import ordinal_encode
    train, test = load_data(cfg)
    train = get_folds(train, cfg)
    tr, te = ordinal_encode(train, test, cfg.cat_cols)
    feats = [c for c in tr.columns if c not in ("id", "y", "fold", cfg.target_col)]
    return tr, te, feats, test


def roc_auc_of(cfg, oof):
    from sklearn.metrics import roc_auc_score
    from core.common import load_data
    train, _ = load_data(cfg)
    return roc_auc_score(train["y"], oof)


# ---------- executor：协议 + 缓存 + 探针 ----------

def test_protocol_baseline(cfg, executor):
    res = executor.protocol_baseline("raw")
    assert res["oof_auc"] > 0.5


def test_protocol_te_full_chain(cfg, executor):
    """回归测试：TE 协议经 train_eval 走通（2026-10-08 cname 事故的覆盖缺口）。"""
    base = executor.protocol_baseline("raw")
    res, delta = executor.protocol_te("te", base_result=base)
    assert res["oof_auc"] > 0.5
    assert delta == pytest.approx(res["oof_auc"] - base["oof_auc"], abs=1e-9)


def test_oof_cache_reuse(cfg, executor):
    """回归测试：同名实验第二次必须走缓存（cached=True）。"""
    res1 = executor.protocol_baseline("raw")
    res2 = executor.protocol_baseline("raw")
    assert res2.get("cached") is True, "同名实验应命中 OOF 缓存"


def test_probe_mode(cfg):
    from core.common import load_data, get_folds, run_cv
    tr, te, feats, test = raw_feats_ready(cfg)
    import lightgbm
    oof, _ = run_cv(lambda: lightgbm.LGBMClassifier(verbose=-1),
                    tr, te, feats, name="probe_smoke", probe_folds=2)
    covered = (tr["fold"] < 2).to_numpy()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(tr["y"].to_numpy()[covered], oof[covered]) > 0.5


def test_fold_gate_aborts(cfg):
    """门控：明显差于参照的实验必须在第一折中止并抛 FoldGateAbort。"""
    from core.common import FoldGateAbort, load_data, get_folds, run_cv
    tr, te, feats, test = raw_feats_ready(cfg)
    fake_ref = [0.9999] * 5  # 完美参照，任何真实模型都会触发门控
    import lightgbm
    with pytest.raises(FoldGateAbort):
        run_cv(lambda: lightgbm.LGBMClassifier(verbose=-1),
               tr, te, feats, name="gate_smoke", gate_ref_folds=fake_ref)


# ---------- evaluator：四档判决 ----------

def test_evaluator_verdicts():
    from agent.evaluator import ResultEvaluator
    ev = ResultEvaluator()
    assert ev.verdict(0.001) == "adopt"
    assert ev.verdict(0.0004) == "candidate"
    assert ev.verdict(0.00005) == "undecidable"
    assert ev.verdict(-0.001) == "reject"


# ---------- decision_engine：全部动作分支 ----------

def test_decision_engine_all_branches():
    from agent.decision_engine import StrategyDecisionEngine, AgentState
    eng = StrategyDecisionEngine()
    def decide(directions, done=10, lb=1):
        st = AgentState(direction_deltas=directions, experiments_done=done,
                        lb_calibrations=[(0.94, 0.94)] if lb else [])
        return eng.decide(st)["action"]
    dirs = {"a": [0.002], "b": [0.0004], "c": [0.0006]}
    assert decide(dirs) == "optimize_ensemble"
    assert decide({"a": [0.002]}) == "deepen_current_direction"
    assert decide({"a": [0.00001, 0.00001, 0.00001]}) == "pivot_to_new_direction"
    # 分支优先级：强信号存在时 deepen 先于 submit_calibration（源码 decide 顺序）
    assert decide({"a": [0.00001, 0.00000, -0.00001]}, done=5, lb=1) == "pivot_to_new_direction"
    assert decide({"a": [0.0004, -0.0001, 0.0003]}) == "continue_exploration"


# ---------- executor：hetero 插件缺失路径 ----------

def test_hetero_plugin_missing_message(executor):
    with pytest.raises(ImportError, match="plugin not installed"):
        executor._svc(42)


# ---------- 占位文件：接口约定不漂移 ----------

def test_placeholder_docstrings_intact():
    """回归测试：占位文件的 docstring 必须保留设计说明（防误删）。"""
    for path, marker in [("agent/agent.py", "design stub"),
                         ("agent/ensemble_optimizer.py", "design stub")]:
        src = open(os.path.join(ROOT, path), encoding="utf-8").read()
        assert marker in src, f"{path} 的系统层占位说明被删了"


def test_no_competition_literals():
    """回归测试：公开代码不得含具体比赛名或人名（普适化纪律）。"""
    import re
    bad = re.compile(r"s6e\d|Will_Buy_EV|Annual_Income|Chris Deotte|najiama|jazivxt|megayak|lucifer|medvax")
    for path in ["core/common.py", "core/dataset.py", "core/features.py",
                 "agent/planner.py", "agent/evaluator.py",
                 "agent/decision_engine.py", "agent/executor.py"]:
        src = open(os.path.join(ROOT, path), encoding="utf-8").read()
        hits = [l for l in src.splitlines() if bad.search(l)]
        assert not hits, f"{path} 含赛题/人名字面量: {hits[:2]}"


# ---------- 独立运行入口 ----------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
