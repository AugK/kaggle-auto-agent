"""EnsembleOptimizer / GreedyMultiViewBlender -- ensemble layer (system layer; design stub).

【Role】blend multiple models' OOF predictions into one stronger submission:
- candidate registry: add_candidate registers each prediction stream (a single
  model or a verified external recipe)
- decorrelation filter: select_diverse drops highly-correlated candidates --
  "diverse but weak adds nothing"; only members of similar strength and low
  correlation bring incremental value
- greedy search: find_optimal_ensemble runs forward greedy weighting over
  probability / percentile-rank / rank-logit views of each member

【Known flaw of greedy】greedy is coordinate ascent -- it only climbs, and gets
trapped in the first basin. Measured: a 29-stream greedy climb once lost to a
hand-written equal-weight average of 4 streams (by 2e-5). The fix is
"meta-member injection": reproduce the external recipe exactly, register it as
one more candidate, let greedy re-climb through that ridge. A judgment-layer
classic, see docs/principles.md.

【Why no implementation here】the blender + orchestration loop are the
"integration artifact": their value is the glue with executor/evaluator/decision
engine and continuous tuning -- that is the system layer (paid). The free layer
teaches the idea and the interface (this docstring + tutorial step 5); the full
implementation ships with the system layer.

【Interface sketch】(signature reference for your own implementation)
    opt = EnsembleOptimizer(y_train)
    opt.add_candidate(name, oof_array)
    sel = opt.select_diverse(names=None)
    result = opt.find_optimal_ensemble(max_models=4)  # members/weights/oof_auc

Sanity gate: when candidates' mean pairwise rank correlation > 0.99, blending
is dead on arrival -- add diversity before blending.
"""