"""ResultEvaluator -- turns an experiment's delta into verdict + insights + risks,
using the noise floor and decision thresholds.
"""
from __future__ import annotations

import numpy as np


class ResultEvaluator:
    def __init__(self, noise_floor: float = 0.0001):
        """
        noise_floor: CV noise floor (std of OOF AUC across full-CV reruns with
        different seeds). Decision thresholds per KB protocol: strong>0.0005,
        medium 0.0003~0.0005, <2*noise_floor undecidable.
        """
        self.noise_floor = noise_floor

    def verdict(self, delta):
        if delta is None:
            return 'no_baseline'
        if delta > 0.0005:
            return 'adopt'          # strong signal
        if delta > 0.0003:
            return 'candidate'      # medium, needs combo re-validation
        if abs(delta) < 2 * self.noise_floor:
            return 'undecidable'    # below 2x noise floor
        return 'reject' if delta < 0 else 'weak_positive'

    def evaluate(self, result: dict, base_auc, tactic: str = '') -> dict:
        delta = None if base_auc is None else result['oof_auc'] - base_auc
        v = self.verdict(delta)
        ev = {
            'experiment': result['name'],
            'tactic': tactic,
            'oof_auc': round(result['oof_auc'], 5),
            'base_auc': None if base_auc is None else round(base_auc, 5),
            'delta': None if delta is None else round(delta, 5),
            'verdict': v,
            'insights': [],
            'risks': [],
        }
        if v == 'adopt':
            ev['insights'].append(f'{tactic or result["name"]} significantly effective (+{delta:.5f}), adopted into best config')
        elif v == 'candidate':
            ev['insights'].append(f'{tactic or result["name"]} weak positive (+{delta:.5f}), needs combo re-validation')
            ev['risks'].append('near noise floor; LB transfer may be invisible')
        elif v == 'undecidable':
            ev['insights'].append(f'{tactic or result["name"]} undecidable (|d|<{2*self.noise_floor:.4f})')
        elif v == 'weak_positive':
            ev['insights'].append(f'{tactic or result["name"]} weak positive (d={delta:+.5f}), kept but not an LB trigger on its own')
        elif v == 'reject':
            ev['insights'].append(f'{tactic or result["name"]} harmful (d={delta:+.5f}), rolled back')
        return ev

    @staticmethod
    def measure_noise_floor(aucs_by_seed: list) -> float:
        """Noise floor from full-CV OOF AUC across seeds (approx: std across seeds)."""
        return float(np.std(aucs_by_seed))
