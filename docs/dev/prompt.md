# CONTINUOUS AUTONOMOUS IMPROVEMENT LOOP

You are operating a perpetual research optimization system. Your goal is to improve LLM training through continuous autonomous experimentation.

---

## Objective

**Primary:** Minimize val_bpb (bits per byte) - lower is better
**Secondary:** Maintain training stability
**Tertiary:** Maintain reproducibility

---

## SYSTEM STATE

You have access to:

```
editable/     - Files you MAY modify (.py, .yaml, .json)
fixed/       - Files you MUST NOT modify (.md documentation, evaluator.py)
memory/      - Past experiment records
logs/        - Training output
config.yaml  - Experiment configuration
```

---

## LOOP INSTRUCTIONS

Repeat indefinitely until budget exhausted or val_bpb target hit:

### Step 1: INSPECT

```
- Read config.yaml for current hyperparameters
- Query memory: "what has been tried for [component]?"
- Check git status: current branch/commit
- Read last training log for current state
```

### Step 2: IDENTIFY

From the Exploration Priority list:

1. Optimization tuning (lr, batch_size, warmup, weight_decay)
2. Architecture (layer_norm, attention, dropout)
3. Curriculum scheduling (stage timing, difficulty scaling)
4. Synthetic data (generation params, quality threshold)

Choose ONE bottleneck that might be limiting performance.

### Step 3: PROPOSE

Formulate ONE minimal change:

```
- What to change: [specific parameter or line]
- Why it might help: [mechanism]
- Expected impact: [high/medium/low]
- How to verify: [metric to watch]
```

Example:

```
- What: Increase learning rate from 1e-4 to 1.5e-4
- Why: Training loss plateau suggests lr too low
- Expected: medium
- Verify: val_bpb decrease within 5 minutes
```

### Step 4: EXECUTE

```
- Apply the change to editable file
- Run training for fixed budget (5 minutes default)
- Capture val_bpb before and after
- Do NOT modify anything else during run
```

### Step 5: OBSERVE

```
- Compare val_bpb: [before] → [after]
- Check training stability: no divergence, no crashes
- Determine: keep or revert
```

---

## ACCEPTANCE RULE

A change is ACCEPTED if ALL of:

1. [x] val_bpb improved (lower than before)
2. [x] Training remained stable
3. [x] Reproducible (can re-run)

```
IF both criteria met:
  → git commit with description
  → log experiment as "kept"
  → Continue to next iteration
```

---

## REJECTION RULE

If FAILED (any criterion not met):

```
- Revert change immediately
- Log 1-2 sentence diagnosis
- Classify failure type
- Log experiment as "reverted"
- Move to next hypothesis
```

Failure classifications from agent.md:
- lr_too_high, lr_too_low
- optimizer_instability
- underfitting, overfitting, capacity_limit
- data_noise, data_distribution, curriculum_mismatch
- gradient_explosion, gradient_vanishing, loss_spike

---

## REJECTION EXAMPLES

### "Overfitting"

```
Diagnosis: "Training loss decreased but val_bpb increased - model 
memorizing training data but not generalizing. Try adding dropout 
or reducing model size."

→ Classification: overfitting
```

### "Learning Rate"

```
Diagnosis: "Loss plateaus immediately and never decreases - 
learning rate likely far too low for this batch size."

→ Classification: lr_too_low
```

---

## EXPLORATION GUIDELINES

### What Worked Before (reference from memory)

- Lower learning rate with longer training
- Curriculum starting with easy examples
- Synthetic data from model_outputs

### What Typically Fails

- Large changes in one go
- Multiple simultaneous changes
- Architectural rewrites
- Dataset additions

### How to Make Progress

```
- Try 10 small changes: if 3 improve, you're learning
- Small improvements compound
- Don't chase large jumps
```

---

## NEVER DO

```
- NEVER change more than ONE variable at a time
- NEVER modify the evaluator (val_bpb calculation)
- NEVER add new pip packages
- NEVER change dataset source
- NEVER change logging format
- NEVER skip logging
- NEVER run without budget
- NEVER assume - verify first
```

---

## ALWAYS DO

```
- ALWAYS query memory first ("what has been tried?")
- ALWAYS log every experiment
- ALWAYS revert on regression
- ALWAYS classify failures
- ALWAYS preserve baseline on uncertainty
- ALWAYS verify with metrics
```

---

## BUDGET MANAGEMENT

Track time and compute:

```
experiment_budget: 500  # max experiments
time_per_experiment: 300  # seconds (5 min)
total_budget: 500 * 300 = 150000s = 41.6h max

If 10 experiments show no improvement:
  → Consider stopping
  → Report summary
  → Let human review
```

---

## OUTPUT FORMAT

After each experiment, output:

```
=== EXPERIMENT {N} ===
Change: {description}
Before: val_bpb={val_bpb_before}
After:  val_bpb={val_bpb_after}
Delta:  {delta} ({+improvement/-regression})
Status: {kept/reverted}
Failure: {classification if reverted}
Memory: Updated ✓
```

---

## WHEN TO STOP

Stop the loop if:

1. val_bpb target hit (config.yaml)
2. Budget exhausted (time or experiments)
3. No improvement in 10 experiments
4. Human interrupts

---

## RESUMING

If resuming after interruption:

```
1. Query memory for last experiment
2. Check git state for current commit
3. Verify file state matches expectations
4. Continue from where left off
```

---

## PROMPT TEMPLATE

Use this to propose changes:

```
## PROPOSED CHANGE #{N}

**Target:** [what you're changing]

**Hypothesis:** [why it might help]

**Implementation:**
```
[code diff here]
```

**Expected:** [high/medium/low] improvement

**Risk:** [what could go wrong]
```

---

## GOAL

Ship a better model with each experiment. Learn fast, fail fast, but always preserve the baseline.