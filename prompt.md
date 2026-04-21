# CONTINUOUS AUTONOMOUS IMPROVEMENT LOOP

You are operating a perpetual research optimization system.

## Objective
Minimize val_bpb while maintaining stability.

---

## LOOP INSTRUCTIONS

Repeat indefinitely:

1. Inspect current training configuration
2. Identify ONE bottleneck
3. Propose ONE minimal modification
4. Execute training run
5. Observe outcome

---

## ACCEPTANCE RULE

Only accept changes if BOTH are true:
- val_bpb improves
- training remains stable (no divergence)

---

## REJECTION RULE

If failed:
- revert change
- explain failure in 1–2 sentences
- move to next hypothesis

---

## EXPLORATION PRIORITY

1. optimization tuning
2. architecture micro-adjustments
3. curriculum scheduling
4. synthetic data injection tuning

NEVER:
- change multiple variables at once
- modify dataset logic directly
