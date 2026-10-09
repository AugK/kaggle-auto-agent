"""CompetitionKnowledgeBase -- the agent's knowledge base.

Three layers (distilled from post-competition retrospectives):
1. universal_principles  hold for any experiment
2. conditional_tactics   applicable conditions + risks attached
3. technique_library     concrete execution protocols and decision rules
4. common_pitfalls       from real mistakes

All code identifiers and structural comments are in English for the open-source community.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Tactic:
    name: str
    condition: str
    check: str                 # how to verify the condition on a dataset
    protocol: str              # key of the executor protocol to use
    parameter_range: dict = field(default_factory=dict)
    risk: str = ''
    evidence: list = field(default_factory=list)   # (competition, delta, note)


@dataclass
class Protocol:
    name: str
    steps: list
    decision_rules: dict


class CompetitionKnowledgeBase:
    def __init__(self):
        self.universal_principles = {
            'proxy_metric': '代理指标(CV)≠真实目标(LB)，每3-5个CV实验做一次LB校准',
            'single_variable': '单变量对照是归因前提：一次只改一个变量（含随机种子）',
            'resolution': '测量分辨率决定结论粒度：差异低于2倍噪声底即不可判定',
            'minimum_cost': '最小成本先测试方向，有效再放大（三折/单种子先行）',
            'noise_floor': '用不同种子重复整套CV测噪声底；低于噪声线的差异不可判定',
            'importance_horizon': '树模型单特征gain系统性低估低增益特征的交互贡献，'
                                  '砍特征决策必须看消融CV而非gain',
        }

        # ---- conditional tactics (validated in prior competitions) ----
        self.conditional_tactics = {
            'target_encoding': Tactic(
                name='target_encoding',
                condition='仅当特征值有大量精确重复（低基数类别或可分箱构造的键）时有效',
                check='column nunique / len(train) 比值低，或分箱后可形成 n>=100 的组',
                protocol='te_protocol',
                parameter_range={'smoothing': [10, 30, 50, 100]},
                risk='目标泄漏：必须严格折内计算(OOF)，测试集用全量训练统计',
                evidence=[
                    ('某表格赛A', +0.00114, '9个TE特征中5个进Top13，最大单项增益'),
                ]),
            'interaction_features': Tactic(
                name='interaction_features',
                condition='特征对有逻辑关系（总量vs分量、同量纲、态度对冲如concern-vs-anxiety）',
                check='领域知识判断特征对的可解释组合；差分+比值双形式交给模型筛选',
                protocol='interaction_protocol',
                parameter_range={},
                risk='高相关交互可能稀释colsample；单特征gain会低估其贡献',
                evidence=[
                    ('某表格赛A', +0.00160, '周期性sin/cos贡献35.7%重要性'),
                    ('某表格赛B', +0.00015, '17个交互/态度特征，弱信号'),
                ]),
            'pseudo_labeling': Tactic(
                name='pseudo_labeling',
                condition='对已强基线常呈中性，对弱基线可能大涨；仅在强基线+严格对齐时尝试',
                check='基线OOF已达平台期（连续实验提升<噪声底）后再试',
                protocol='pseudo_label_protocol',
                parameter_range={'t1': [0.9], 't0': [0.03], 'weight': [0.5]},
                risk='过拟合；按mask拼接的数据必须逐行验证标签对齐（S6E8曾因未对齐误判）',
                evidence=[
                    ('某表格赛A', 0.0, '修复标签对齐bug后精确中性'),
                ]),
            'feature_pruning': Tactic(
                name='feature_pruning',
                condition='仅当消融CV证明无损时执行；gain阈值不可作为砍特征依据',
                check='对候选删除集做三折消融，delta >= -噪声底 才可删',
                protocol='ablation_protocol',
                parameter_range={},
                risk='an aggressive gain-based prune once cost -0.0017~-0.0020',
                evidence=[
                    ('某表格赛A', -0.00180, 'gain<1%全删23特征方案，已中止'),
                ]),
            'seed_ensembling': Tactic(
                name='seed_ensembling',
                condition='通用低成本增益，任何模型都适用',
                check='无需条件，但增益量级约+0.0001量级（噪声底边缘）',
                protocol='seed_ensemble_protocol',
                parameter_range={'seeds': [[42, 123, 456]]},
                risk='接近噪声底，LB验证收益可能不可见',
                evidence=[
                    ('某表格赛A', +0.00007, '3种子等权，OOF可测LB边缘'),
                ]),
        }

        # ---- technique library (execution protocols) ----
        self.technique_library = {
            'te_protocol': Protocol(
                name='te_protocol',
                steps=[
                    '1. 基线建立（同fold同种子CV）',
                    '2. 逐个TE验证（OOF折内、smoothing=30起步）',
                    '3. Smoothing调优 {10,30,50,100}',
                    '4. 组合TE（类别×类别键）泄漏检查',
                    '5. 最终组合验证（单变量对照）',
                ],
                decision_rules={
                    'strong_signal': 'delta > 0.0005 → 立即采用',
                    'medium_signal': '0.0003 < delta < 0.0005 → 候选，需组合复验',
                    'weak_signal': 'delta < 0.0002 → 不可判定（低于2倍噪声底）',
                }),
            'interaction_protocol': Protocol(
                name='interaction_protocol',
                steps=[
                    '1. 列出有逻辑关系的特征对',
                    '2. 差分+比值+乘积三形式生成',
                    '3. 整组加入做单变量对照（不逐个删，防gain误判）',
                    '4. 有效则整组保留，无效则整组弃',
                ],
                decision_rules={'adopt': '整组 delta >= 0.0003 且各折方向一致'}),
            'pseudo_label_protocol': Protocol(
                name='pseudo_label_protocol',
                steps=[
                    '1. 用最强submission做种子',
                    '2. 高置信度阈值 T1=0.9 / T0=0.03',
                    '3. 伪标签样本权重0.5',
                    '4. 逐行验证标签对齐后才能训练',
                ],
                decision_rules={'adopt': 'OOF delta > 0.0005 才上LB'}),
            'seed_ensemble_protocol': Protocol(
                name='seed_ensemble_protocol',
                steps=['1. 同配置换种子{42,123,456}', '2. OOF等权平均', '3. 检查OOF增益'],
                decision_rules={'adopt': 'OOF delta > 0（免费增益）'}),
            'ablation_protocol': Protocol(
                name='ablation_protocol',
                steps=['1. 全量基线', '2. 删除候选集三折CV', '3. delta < 噪声底才可删'],
                decision_rules={'keep': '三折delta <= -噪声底 → 不可删'}),
        }

        self.common_pitfalls = {
            'data_leakage': '目标编码必须在折内计算，否则目标泄漏（CV虚高LB崩）',
            'single_importance': 'gain只砍底不选顶：gain系统性低估低增益特征交互贡献',
            'measurement_noise': '差异低于测量分辨率的结论不可判定，不要过度解读',
            'pandas_traps': 'pandas>=3.0的astype(str)保留NA，需显式fillna后再编码',
            'label_alignment': '按mask拼接的数据必须逐行验证标签对齐',
            'eval_protocol': '50%简单分割的评估不可靠，必须用冻结fold的多折CV',
            'bg_jobs': '启动新任务前先清点后台任务，重复任务会抢CPU拖慢一切',
        }

        # validated experiment records (referenced by the decision engine)
        self.verified_results = []

    # ---- 查询接口 ----
    def query_applicable_tactics(self, data_analysis: dict) -> list:
        """Return applicable tactics for a data analysis (with reasons)."""
        applicable = []
        n_train = data_analysis.get('n_train', 0)
        for tac in self.conditional_tactics.values():
            if tac.name == 'target_encoding':
                low_card = [c for c, k in data_analysis.get('cardinality', {}).items()
                            if k <= max(50, n_train * 0.001)]
                if low_card or data_analysis.get('binable_cols'):
                    applicable.append((tac, f'低基数列: {low_card or data_analysis["binable_cols"]}'))
            elif tac.name == 'interaction_features':
                if data_analysis.get('n_numeric', 0) + data_analysis.get('n_cat', 0) >= 6:
                    applicable.append((tac, '特征数足够，存在可组合的领域关系'))
            elif tac.name == 'seed_ensembling':
                applicable.append((tac, '通用低成本增益'))
            elif tac.name == 'pseudo_labeling':
                applicable.append((tac, '仅在基线平台期后考虑'))
        return applicable

    def get_protocol(self, key: str) -> Protocol:
        return self.technique_library[key]

    def check_pitfalls(self, technique: str) -> list:
        keys = {
            'target_encoding': ['data_leakage', 'pandas_traps'],
            'feature_pruning': ['single_importance'],
            'pseudo_labeling': ['label_alignment', 'data_leakage'],
        }
        return [self.common_pitfalls[k] for k in keys.get(technique, [])]

    def ingest_result(self, name: str, tactic: str, delta: float, verdict: str):
        """Record a validated experiment for the decision engine and future competitions."""
        self.verified_results.append({
            'experiment': name, 'tactic': tactic,
            'delta': round(delta, 5), 'verdict': verdict,
        })

    def summary(self) -> str:
        lines = ['== Universal Principles ==']
        lines += [f'  {k}: {v}' for k, v in self.universal_principles.items()]
        lines.append('== Conditional Tactics ==')
        lines += [f'  {t.name}: {t.condition} | risk: {t.risk}'
                  for t in self.conditional_tactics.values()]
        lines.append('== Verified Results ==')
        lines += [f"  {r['experiment']}: delta={r['delta']:+.5f} [{r['verdict']}]"
                  for r in self.verified_results]
        return '\n'.join(lines)
