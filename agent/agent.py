"""KaggleAutoAgent -- the orchestration loop (system layer; this file is a design stub).

【Role】chains the six modules into one loop:
load data -> query KB for tactics -> planner schedules -> executor runs ->
evaluator rules -> decision engine picks the next move -> (when effective
directions >= 3) ensemble optimization -> persist state -> repeat.

【Design rationale (judgment layer)】
- State lives outside the process: experiment history, per-direction deltas and
  CV-LB calibration points go to logs/agent_state.json -- a crashed run loses
  nothing, and retrospectives have a raw ledger to read.
- Decision thresholds are explicit: what counts as a strong signal (+0.001),
  a stall (3 consecutive below noise floor), when to stop exploring and start
  blending -- all constants in the decision engine. Changing a threshold is
  changing strategy.
- Human-machine boundary: the agent executes and records. Which competition to
  enter, when to overrule its plan, when to submit -- always a person. The
  illusion of full-auto is more dangerous than honest semi-auto.

【Why no implementation here】the orchestration loop is the glue of the six
modules -- simple in shape, expensive in the scars it needs (cache coherence,
probe/full-mode handoff, failure recovery). Assembly and maintenance belong to
the system layer; the free tutorial (docs/tutorial.md five steps) is enough to
write your own version with an AI assistant -- that is the intended usage.

【Reference shape】
    PYTHONPATH=. python -m agent.agent --status   # current state + suggestion
    PYTHONPATH=. python -m agent.agent            # full pipeline (skips cached)
"""