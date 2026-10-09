"""StrategyDecisionEngine -- decides the next move from current state
(best config, per-direction delta history).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentState:
    best_auc: float = 0.0
    best_config: str = ''
    direction_deltas: dict = field(default_factory=dict)   # tactic -> [delta,...]
    lb_calibrations: list = field(default_factory=list)    # [(cv, lb), ...]
    experiments_done: int = 0
    submitted: list = field(default_factory=list)


class StrategyDecisionEngine:
    # decision thresholds (from KB protocol)
    STRONG = 0.001
    STALL_RATE = 0.0002
    MIN_DIRECTIONS_FOR_ENSEMBLE = 3

    def decide(self, state: AgentState) -> dict:
        strong_dirs = [t for t, ds in state.direction_deltas.items()
                       if ds and max(ds) > self.STRONG]
        effective_dirs = [t for t, ds in state.direction_deltas.items()
                          if ds and max(ds) > 0.0003]
        recent = [d for ds in state.direction_deltas.values() for d in ds[-3:]]
        stalled = recent and all(abs(d) < self.STALL_RATE for d in recent)

        if len(effective_dirs) >= self.MIN_DIRECTIONS_FOR_ENSEMBLE:
            return {'action': 'optimize_ensemble',
                    'rationale': f'{len(effective_dirs)}effective directions >= 3; ensemble beats more single-point probes',
                    'payload': {'directions': effective_dirs}}
        if strong_dirs:
            return {'action': 'deepen_current_direction',
                    'rationale': f'strong directions {strong_dirs}; keep deepening (tuning/combos)',
                    'payload': {'directions': strong_dirs}}
        if stalled:
            return {'action': 'pivot_to_new_direction',
                    'rationale': 'last 3 experiments below noise floor; pivot (model matrix/ensemble/pseudo-labels)',
                    'payload': {}}
        if not state.lb_calibrations and state.experiments_done >= 3:
            return {'action': 'submit_calibration',
                    'rationale': '3 CV experiments and no LB calibration yet; submit to establish CV-LB mapping',
                    'payload': {'config': state.best_config}}
        return {'action': 'continue_exploration',
                'rationale': 'still in productive exploration; continue the current plan', 'payload': {}}
