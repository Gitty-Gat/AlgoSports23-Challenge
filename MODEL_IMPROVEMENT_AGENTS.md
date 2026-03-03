
---

# MODEL_IMPROVEMENT_AGENTS.md

---

# ALGO SPORTS 23 – MODEL IMPROVEMENT GOVERNANCE DOCUMENT

### Codex Agent Hard-Contract Specification

**Model: GPT-5.3 Codex (RE = Extra High)**
**Authority Level: Implementation + Refactor + Full Pipeline Control**

---

# I. VERIFIED CURRENT STATE (“HARD TRUTH”)

This section defines the non-negotiable baseline of the project at the start of this improvement cycle.

Codex MUST treat this as ground truth.

---

## 1. Current Leaderboard Position

* Current submission RMSE: **262.67280**
* Current rank: **9th / 16**
* 1st place RMSE: **249.98799**
* Gap to 1st: **12.68481 RMSE**

This gap is material.
Incremental tweaks will not close it.
Structural signal upgrades are required.

---

## 2. Current Model Architecture (Verified From Artifacts)

### Base Features (F2 only)

```
["massey_diff",
 "elok_diff",
 "games_played_diff",
 "volatility_diff"]
```

### Base Model

* ElasticNet(alpha=1.0, l1_ratio=0.12)

### Postprocess

* expand_affine_global
* slope s = 1.4269
* cap not hit

### Observed Diagnostics

| Metric               | Current |
| -------------------- | ------- |
| mean_rmse            | 35.9447 |
| mean_tail_dispersion | 0.4806  |
| mean_tail_bias       | -9.4157 |
| fold2_tail_bias      | ≈ -20.7 |
| cap_hit_rate         | 0.25    |
| corr(y_true, y_pred) | ~0.55   |

---

## 3. Primary Failure Modes (Confirmed)

### A. Tail Compression

Predicted margins have <50% of true dispersion in top 20% games.

### B. Negative Tail Bias

Large underprediction in blowouts.

### C. Weak Correlation Ceiling (~0.55)

Scaling cannot fix correlation.

### D. Model Collapse

Top-3 configs are minor ElasticNet tweaks.
Search space lacks diversity.

### E. Fold 2 Failure (Mar–Apr)

Largest bias and instability.

---

# II. GLOBAL RULES FOR CODEX

Codex must adhere to the following constraints:

---

## 1. NO MINIMAL SOLUTIONS

Codex is forbidden from:

* Making micro-parameter nudges only
* Repeating prior architectures
* Skipping diversity in model families
* Stopping early without metrics being achieved
* Claiming improvement without evidence artifacts

---

## 2. HARD METRIC CONTRACTS

Each phase defines required metric gates.
Codex CANNOT advance to the next phase until ALL gates are satisfied.

If gates fail:

* Codex must iterate within that phase
* Expand search space
* Adjust modeling choices
* Re-run until metrics pass

No partial credit.

---

## 3. Submission File Governance

After each phase completion:

### A. File Rotation Rule

If `Submission.zip` already exists:

* Rename to `Submission.zip<NUM>`
* Where NUM = most recent model number

Example:

* Current submission is #11
* Codex finishes Phase 1
* Rename:

  * `Submission.zip` → `Submission.zip11`
  * Save new results as `Submission.zip`

---

### B. Submission.zip Must Contain

```
Predictions.csv
Rankings.xlsx
```

Both must reflect latest model.

---

### C. Reports

Codex must:

1. Generate:

   * `final_report.pdf`
   * `run_report.md`

2. Create subfolder:

   ```
   legacy_reports/
   ```

3. Move prior:

   * `final_report.pdf`
   * `run_report.md`

   into:

   ```
   legacy_reports/
   ```

4. Rename with model number:

   ```
   final_report_model_11.pdf
   run_report_model_11.md
   ```

5. Save new reports in root.

---

## 4. Evidence Requirement

At the end of each phase Codex must provide:

* Full metric table
* Fold-by-fold RMSE
* Tail dispersion
* Tail bias
* Correlation
* Cap-hit rate
* Model architecture summary
* Feature list
* Hyperparameters
* Search space explored
* Number of fits
* Justification of improvements

No summary without proof.

---

# III. PHASE STRUCTURE

---

# PHASE 1

### (Run 1 + Run 2)

### Objective: Break the 260 Plateau Structurally

---

## Phase 1 Structural Changes Required

### Promote nextgen_pipeline as default engine

Must include:

* Recency EMAs
* Opponent-adjusted margins
* Off/Def signals
* Conference strength
* Trend features
* Knot features

F2-only modeling is no longer allowed.

---

### Strong Model Bracket Required

Codex must run:

* Ridge
* HuberRegressor
* HistGradientBoostingRegressor
* GradientBoostingRegressor

ElasticNet allowed only as baseline comparator.

---

### Relax Regime Gate

Modify blowout gate:

OLD:

```
tail_dispersion < 0.85 AND tail_improve < 2.0 → invalid
```

NEW:

```
tail_dispersion < 0.60 AND tail_improve < 1.0 → invalid
```

Regime models must be allowed to compete.

---

### Expand ETA grid

Add:

```
0.75
0.80
```

---

## PHASE 1 HARD METRIC GATES

Codex CANNOT proceed until ALL are met:

| Metric               | Required |
| -------------------- | -------- |
| mean_rmse            | ≤ 35.2   |
| corr                 | ≥ 0.60   |
| mean_tail_dispersion | ≥ 0.58   |
| mean_tail_bias       | ≥ -6.0   |
| fold2_rmse           | ≤ 38.0   |

If not met:

* Expand feature sets
* Expand model complexity
* Expand hyperparameter search
* Re-run

No advancement allowed.

---

# PHASE 2

### (Run 3 + Run 4)

### Objective: Repair Tails + Fold 2 Instability

---

## Phase 2 Structural Requirements

### Turn On Selection Penalties

Must include non-zero:

```
lambda_cap_hit ∈ {0.05, 0.10}
lambda_tbias ∈ {0.002, 0.005}
```

---

### Enable Regime + Heterosk Modules

Must include:

* expand_regime_affine_v2
* expand_regime_affine_v2_heterosk

---

### Feature Family Isolation Experiments

Must evaluate:

* Recency-only profile
* OppAdj-only profile
* Conference-only profile

Measure fold 2 deltas.

---

## PHASE 2 HARD METRIC GATES

All must pass:

| Metric               | Required |
| -------------------- | -------- |
| mean_rmse            | ≤ 34.7   |
| corr                 | ≥ 0.62   |
| mean_tail_dispersion | ≥ 0.65   |
| mean_tail_bias       | ≥ -4.0   |
| fold2_rmse           | ≤ 37.0   |
| cap_hit_rate         | ≤ 0.05   |

If not met:

* Increase search space
* Adjust rating dynamics
* Iterate

No advancement allowed.

---

# PHASE 3

### (Run 5)

### Objective: 1st Place Attempt

---

## Phase 3 Required Actions

### Elo Dynamics Sweep

Grid:

* elo_k ∈ {16, 24, 32, 40}
* elo_home_adv ∈ {40, 60, 80}
* elo_decay_a ∈ {0.15, 0.25, 0.35}
* elo_decay_g ∈ {25, 50, 100}

---

### Full Model + Postprocess Search

Must include:

* All strong learners
* All regime modules
* Full eta grid
* Full penalty grid
* Expanded hyperparameter grids

---

### Increase Budget

```
ALGOSPORTS_MAX_MODEL_FITS = 1500+
ALGOSPORTS_MAX_TOTAL_SECONDS = 1800+
```

Must actually use > 1000 fits.

---

## PHASE 3 HARD METRIC GATES

| Metric               | Required |
| -------------------- | -------- |
| mean_rmse            | ≤ 34.2   |
| corr                 | ≥ 0.64   |
| mean_tail_dispersion | ≥ 0.70   |
| mean_tail_bias       | ≥ -2.5   |
| fold1_rmse           | ≤ 36.5   |
| fold2_rmse           | ≤ 36.5   |

If not met:

* Continue search
* Expand model capacity
* Refine features

No stopping early.

---

# IV. ANTI-CHEAT CLAUSE

Codex is forbidden from:

* Hardcoding predictions
* Using leakage
* Using leaderboard feedback inside CV
* Skipping folds
* Artificially inflating dispersion without CV validation
* Claiming improvement without updated artifacts

All improvements must be OOF-validated.

---

# V. COMPLETION CRITERIA

The improvement cycle is complete only when:

* Phase 3 gates are met
* Submission.zip is produced
* Reports saved and archived
* Full evidence summary generated
* All governance rules followed

---

# END OF MODEL_IMPROVEMENT_AGENTS.md

---
