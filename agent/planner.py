"""ExperimentPlanner -- staged experiment planning.

From data analysis and the knowledge base, generate a phased plan:
baseline(30m) -> feature_engineering(2-4h) -> model_optimization(4-8h) -> ensemble(2-4h)
"""
from __future__ import annotations

from agent.knowledge_base import CompetitionKnowledgeBase


class ExperimentPlanner:
    def __init__(self, kb: CompetitionKnowledgeBase):
        self.kb = kb

    def plan_competition_strategy(self, data_analysis: dict, competition_info: dict) -> list:
        phases = []

        # Phase 1: quick baseline
        phases.append({
            'phase': 'baseline', 'time_budget': '30min', 'priority': 'must-complete',
            'experiments': [
                {'protocol': 'protocol_baseline', 'featset': 'raw',
                 'goal': 'get a working baseline, freeze folds'},
            ]})

        # Phase 2: feature engineering (tactics filtered by KB applicability)
        applicable = self.kb.query_applicable_tactics(data_analysis)
        fe_exps = []
        for tac, reason in applicable:
            if tac.name == 'interaction_features':
                fe_exps.append({'protocol': 'protocol_interaction', 'featset': 'inter',
                                'goal': f'interaction group A/B ({reason})'})
            if tac.name == 'target_encoding':
                fe_exps.append({'protocol': 'protocol_te', 'featset': 'te',
                                'goal': f'TE 5-step protocol ({reason})'})
            if tac.name == 'target_encoding':
                fe_exps.append({'protocol': 'protocol_te', 'featset': 'combo',
                                'goal': 'TE + interactions + combo keys, full set'})
        phases.append({'phase': 'feature_engineering', 'time_budget': '2-4h',
                       'experiments': fe_exps})

        # Phase 3: model optimization (model matrix, single seed; multi-seed
        # ensembling intentionally skipped per user decision)
        phases.append({'phase': 'model_optimization', 'time_budget': '4-8h',
                       'experiments': [
                           {'protocol': 'model_matrix', 'featset': 'BEST',
                            'goal': 'LGB/XGB/CatBoost on same features (seed 42)'},
                       ]})

        # Phase 4: ensemble optimization
        phases.append({'phase': 'ensemble', 'time_budget': '2-4h',
                       'experiments': [
                           {'protocol': 'optimize_ensemble',
                            'goal': 'rank-correlation decorrelation + weight grid'},
                       ]})
        return phases

    def plan_v2(self):
        """Cross-TE catch-up plan (single-seed constraint), see docs/tutorial.md."""
        return [
            {'phase': 'v2_features_A', 'time_budget': '3-5h', 'source': 'nb1 V5/V6/V7',
             'experiments': [
                 {'protocol': 'protocol_v2_batch', 'kind': 'lgb', 'featset': f'v2_{b}',
                  'goal': goal}
                 for b, goal in [('a1', 'multi-scale qbin TE + cat/catpair TE (main gain)'),
                                 ('a2', 'cross-key TE quadruples (main gain)'),
                                 ('a3', 'exact-value TE'),
                                 ('a4', 'combo frequency + segmented one-hot'),
                                 ('a5', 'Fourier / local basis / forensics features'),
                                 ('a6', 'hand-crafted prior score')]]},
            {'phase': 'v2_models_B', 'time_budget': '2-4h',
             'experiments': [
                 {'protocol': 'protocol_v2_batch', 'kind': 'cat', 'featset': 'BEST',
                  'goal': 'CatBoost, current best single model'},
                 {'protocol': 'protocol_v2_batch', 'kind': 'xgb_nb1', 'featset': 'BEST',
                  'goal': 'XGB nb1 recipe depth6/mcw30/l8/lr0.01'},
                 {'protocol': 'protocol_v2_batch', 'kind': 'hgbc', 'featset': 'BEST',
                  'goal': 'HistGradientBoosting nb2 recipe'}]},
            {'phase': 'v2_blend_C', 'time_budget': '1-2h',
             'experiments': [
                 {'protocol': 'optimize_ensemble_v2',
                  'goal': 'three views prob/rank/logit + greedy alpha blend (80 steps)'},]},
        ]
