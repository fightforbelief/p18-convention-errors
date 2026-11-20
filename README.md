# P18 direction review and pilot (2026-11-19)

## 1. Verdict on the original P18

The original P18 is "Verified Agentic DS Copilot" (goals_archived.pdf, p.15). An agent does data audit, model selection and experiment planning, and its output is checked by schema, statistical checks, unit tests and reproducibility constraints. Planned deliverables: tool, evaluation dataset and reliability report.

**Verdict: do not do it as written.**

1. **The space is crowded as of Nov 2026.** Directly overlapping work:
   - RADAR, "Fail Loudly" (arXiv 2609.32528): an auditable runtime with a five-way silent-error taxonomy, 1,054 tasks.
   - "Sanity Checks for Agentic Data Science" (2604.11003): null datasets.
   - "Towards Reliable AI Data Scientists" (2609.35255): workflow harnesses.
   - "When Tools Silently Lie" (2609.37153).
   - `ds-agent-harness`: a verification gate with about 4,100 pre-registered runs. It already reports that the gate can turn right answers into wrong ones.
   - "Many AI Analysts, One Dataset" (2602.18710) and "The Agentic Garden of Forking Paths" (2607.01507).
   - Older benchmarks: DSBench, MLE-bench, DS-Agent, Data Interpreter.
2. **The deliverable is a system, not a finding.** "Tool + eval set + reliability report" is the kind of AI-pipeline paper the PDF itself warns against.
3. **There is no sharp research question.** The PDF's own question ("when should the agent trust itself, verify, defer") overlaps P8 and P21.

## 2. Proposed reframe: convention errors in panel pipelines

**Question.** Some panel-data processing errors are locally correct: each line does what it says, the code runs, and schema and distribution checks pass. Yet they silently change *who* is in the sample.

- Do standard checks (including LLM code review) catch these errors?
- Do they bias the conclusion?
- What kind of check does catch them?

**Why it is not already covered.** It starts from a real incident: the P12 finding. FTSFR's `QE -> 3mo` grid step drops about 35% of post-1976 bank entrants and reverses model rankings. That bug survived an elaborate audit pipeline.

The class of error is specific: calendar, frequency and life-cycle conventions in longitudinal data. RADAR-style checks look at one operation at a time, so they cannot see these errors, because every individual operation is correct.

**Proposed mechanism.** Composition errors show up as *non-uniform retention along entity life-cycle coordinates*: entry calendar, age since entry, and exit status. A raw-vs-processed entity-period ledger (provenance plus retention uniformity) is cheap, label-free and model-agnostic.

**Hypotheses for the full study (to pre-register):**

- **H1.** Schema and distribution checks pass on most convention bugs.
- **H2.** Undetected bugs move the headline estimate beyond its CI, and the bias falls on a non-random subpopulation.
- **H3.** LLM reviewers, given code plus run summary, detect convention bugs at a low rate. Detection rises when the ledger output is supplied.
- **H4.** The ledger detects held-out bugs with a low false-alarm rate. The held-out bugs must not be designed by us: they would come from LLM-written pipelines, GitHub issues, and real benchmark repos such as FTSFR.

## 3. Pilot v0 (`pilot/pilot_v0.py`): synthetic panel, known structure

Setup:
- Panel: 4,000 banks, quarterly, 1990Q1–2020Q4.
- Incumbents mean-revert; entrants are near unit root; 30% of banks exit.
- Conclusion measured: mean MASE(Naive) − MASE(HistoricAverage), h = 4.

Estimate shift relative to the clean pipeline, over three seeds:

| Pipeline | Shift vs clean | Schema check | Stats check | Ledger |
|---|---|---|---|---|
| B1 grid step `first + DateOffset(months=3k)` (P12 bug) | +41…+44% | pass | pass | **flags** |
| B2 attributes inner-joined from final snapshot (survivorship) | −9…−11% | pass | pass (2/3) | **flags** |
| B3 balanced panel + ffill | −44…−53% | pass | pass | **flags** |
| B4 lag/rolling features then `dropna()` | +8…+11% | pass | pass | **flags** |
| B5 MASE scale computed on train+test (non-composition control) | −1% | pass | pass | passes (expected miss) |
| Clean | 0 | pass | pass | passes (no false alarm) |

- **B1 reproduces the P12 mechanism exactly.** Banks first observed on 30 Jun or 30 Sep land on 30 Dec and are dropped. The evaluated entrant share falls from 0.49 to 0.33, and the CI does not overlap the clean CI.
- **No winner flip yet.** That is a property of this DGP's large clean gap. The paper's metric should be bias relative to the CI, not winner flips.
- **Caveat: the result is circular.** The ledger's life-cycle coordinates were chosen while looking at B1–B4. H4 must be tested on bugs we did not write.

LLM reviewer probe (`pilot/llm_review.py`): prompts are generated (`pilot/prompt_*.txt`), but the run is blocked because the `claude` CLI is not logged in on this machine.

## 3b. Held-out bugs vs the frozen ledger (`pilot/heldout_v0.py`)

The ledger was frozen by SHA-256 before any held-out bug was written (`FREEZE_LEDGER.txt`). Each held-out bug is a pattern documented elsewhere:

- pandas offset anchoring (issue #9586; `QuarterEnd` n=1 rolls forward)
- current-constituent survivorship
- CCM-style many-to-many links
- time-series leakage patterns from Kapoor & Narayanan

A bug counts as "harmful" if its estimate leaves the clean pipeline's 95% CI, so harm is not decided by our judgment. Synthetic panel, 3 seeds:

| Held-out pipeline | Shift | Harmful | Schema | Stats | Ledger |
|---|---|---|---|---|---|
| x1 90-day step grid | panel ~empty | yes | – | **flags** | **misses** |
| x2 `date + QuarterEnd()` roll-forward | 0 | no | – | – | flags (provenance) |
| x3 keep banks active in last year | −11% | **yes** | – | – | **flags** |
| x4 undisclosed ≥40-quarter history filter | +5% | no | – | – | flags (composition) |
| x5 `shift(-1)` target then `dropna` | +2% | no | – | – | – |
| x6 many-to-many link, stale copy kept | 0 | no | – | – | flags (provenance) |
| x7 pooled winsorize 1/99 (benign) | −3% | no | – | – | **false alarm** |
| x8 global z-score before split | 0 | no | – | flags | flags |
| x9 shuffle rows / id round-trip (benign) | 0 | no | – | – | – |

What this says:

1. **The ledger fails on a near-empty panel.** On x1, every cohort's retention collapses to about 0, which the ledger reads as "uniform", and provenance is vacuously true. Fix for v1: report absolute retention and floor it. This is a real flaw, found only because the bugs were held out.
2. **Value provenance is the wrong default.** It fires on legitimate target transforms such as winsorizing. For v1, split it into key provenance (always on) and value provenance (opt-in).
3. **x3 is the only harmful bug that no baseline check catches, and the ledger catches it.** This is the same survivorship pattern as P12, which is weak but consistent support for H1 and H4.
4. **In this task, "harmful" is narrow.** MASE is scale-invariant and the models are univariate, so many real errors (x2, x6, x8) don't move this particular estimate. The paper needs ≥2 downstream tasks: forecasting plus a cross-sectional regression.

## 3c. Real data: FTSFR Call Report panels (`pilot/realdata_v0.py`)

Data: P12 confirm-run panels (`panel.parquet`), 4,000-bank random subsample, 1976Q1–2020Q1.

Deviation D1: on real data, B1 uses a 60-year grid horizon instead of the frozen 40 years, because the real span is 45 years and the frozen cap would have truncated incumbents.

**Ledger v0 failed on bank_leverage.** It flagged all 14 pipelines, including the benign reshuffle. The clean pipeline's own quality filter already gives a retention spread of 0.135 against raw, so "uniform vs raw" is the wrong rule.

**Ledger v1 (`pilot/ledger_v1.py`) replaces it.** It is differential against a declared reference pipeline, adds an absolute retention floor, and checks keys only for provenance (values are not checked). It was frozen with written predictions *before* it touched bhc_leverage (`FREEZE_LEDGER.txt`).

**Pre-registered test on bhc_leverage: passed.** 8/8 predicted flags fired and 0/6 predicted non-flags fired (criterion: ≥7/8 and ≤1/6).

Real-data results (bhc_leverage / bank_leverage):

| Pipeline | Shift in headline | Harmful (outside clean CI) | Stats check | Ledger v1 |
|---|---|---|---|---|
| B1 P12 grid step | +0.3% / −2.7% | no / no | flags / **misses** | flags |
| B2 survivorship join | −26% / −20% | yes / yes | flags | flags |
| B3 ffill | −66% / −95% | yes / yes | flags | flags (R1 keys) |
| B4 warm-up dropna | +1% / +6% | no / yes | **misses** | flags |
| x3 active-at-end | −27% / −18% | yes / yes | flags | flags |
| x4 ≥40q history filter | −16% / −5% | yes / yes | **misses** | flags |
| B5 MASE scale leak | +8% / +10% | yes / yes | misses | **misses** (out of scope) |
| x5 `shift(-1)` + dropna | +7% / +5% | yes / yes | misses | **misses** (window shift, not composition) |
| x6 / x7 / x8 / x9 | ~0 | no | x8 only | no flag (correct) |

What the real data adds:

1. **Silent, harmful composition bugs exist on real bank panels.** x4 shifts the estimate by −16% and B4 by +6%. Both pass schema and distribution checks, and v1 flags them.
2. **A composition change does not automatically bias the result.** B1 (the P12 bug) cuts the evaluated entrant share from 0.87 to 0.78 (bhc) and 0.32 to 0.21 (bank), yet barely moves the *leverage* headline. In P12 *cash liquidity*, the same bug reversed rankings. This points to the paper's theory: bias ≈ composition shift × outcome heterogeneity across life-cycle cohorts. The ledger measures the first factor; the second has to be estimated. That turns the ledger from an alarm into a decomposition.
3. **Scope boundary.** v1 does not catch metric leakage (B5) or evaluation-window shifts (x5). The paper should claim composition errors only, and pair the ledger with existing leakage tooling.
4. **Open caveats:**
   - On bhc_leverage only the *data* was held out; the bug families were the same.
   - New bug families are still needed, from LLM-written pipelines and mined GitHub issues.
   - The LLM-reviewer arm (H3) has not run (the CLI is not logged in).

## 3d. Bias decomposition (`pilot/decomp.py`, frozen before the test panels)

Write Δ for the headline (mean per-bank MASE(Naive) − MASE(HA)). Cells are the raw life-cycle cohorts (entry × entry quarter × exit status). For a pipeline p against the reference r:

ΔΔ = **T1** between-cohort composition + **T2** within-cohort selection + **T3** series alteration (exact identity).

**Result 1 (pre-registered D2, passed on both test panels).** Pure-selection bugs (B2, x3, x4, and B1) have T3 = 0 to numerical precision. Their whole effect equals T1+T2, which can be computed from the reference run's per-bank results plus the candidate's retained-bank set, *without refitting any model*. Alteration bugs (B3 ffill, B5 scale leak) have all of their effect in T3, which the ledger cannot see.

So a composition change's effect on the conclusion can be audited for free, and the two failure modes separate cleanly.

**Result 2 (pre-registered D1 and D3, failed).** The P12 grid bug B1 is *not* harmful to the aggregate on any of 4 real panels:

| Panel | Cohort TV (B1) | Harm ratio (B1) |
|---|---|---|
| bhc_cash | 0.41 | 0.51 |
| bank_cash | 0.17 | 0.04 |
| bhc_lev | 0.41 | 0.07 |
| bank_lev | 0.15 | 0.57 |

B1 moves a lot of composition, but it selects on *calendar quarter of entry*, and outcomes barely vary along that coordinate. The harmful bugs select on outcome-correlated coordinates:

| Bug | Selects on | Harm ratio |
|---|---|---|
| B2, x3 | exit status (survivorship) | 1.7–5.5 |
| x4 | history length | 1.1–3.3 |

This matches P12's own record: the grid bug changed MASE by only 0.2–1.3%, and excluded entrants behaved like retained ones. My D1 prediction ignored that record, which was a mistake.

**What this means for the paper.** The finding is more useful than "silent bugs bias results":

1. Composition-changing bugs split into *coverage* errors and *bias* errors. Coverage errors select on outcome-irrelevant coordinates, such as the P12 grid bug. They shrink the population a claim covers (35% of entrants gone) but leave the estimate intact. Bias errors select on outcome-correlated coordinates, such as survivorship and history filters.
2. Schema and distribution checks cannot tell them apart, and neither can the ledger's retention spreads. The decomposition (T1+T2, computed for free) can.
3. For P12, the grid bug is a *coverage* problem. The paper should say "the benchmark's conclusions silently exclude a third of entrants", not "the benchmark is biased".

## 3e. Full panels × three estimands (`pilot/estimands.py`, frozen before the run)

**Setup.**
- Data: all four panels at full size. bank_leverage has 22,965 banks, bank_cash 23,862, bhc_leverage 13,761, bhc_cash 13,770.
- Estimands:
  - **E1:** mean ΔMASE(Naive − HA), as before.
  - **E2:** fixed-effects AR(1) persistence ρ (an Sxx-weighted mean of per-bank ρ).
  - **E3:** the ranking of six forecasters.
- Decomposition v2 handles any weighted-mean estimand. The ranking is decomposed as 15 pairwise gaps.
- Material (does not depend on n): the shift is ≥ 10% of the E1 effect, or ≥ 10% of (1 − ρ) for E2, or the E3 top-1 model changes.
- Results: `pilot/estimands_*_full.csv`. Grader: `pilot/grade_v2.py`.

**Grading of the six pre-registered predictions: 4 pass, 2 narrowly fail.**

| | Prediction | Result |
|---|---|---|
| P1 | Selection bugs: T3 = 0 on E1 and E2; the ranking forecast with no refit is exact | **pass** (all cells; ranking 16/16) |
| P2 | B1 harmful under the CI rule on full bank_leverage but not material | **fail.** Harm 0.90. Not material on 4/4 panels (shift 0.8–2.6%) |
| P3 | B2 and x3 material on E1, 4/4 panels | **pass** (15.9–23.8%) |
| P4 | x4 material on E2 for ≥ 3 panels; E2 shift > 0 | **fail.** Cash panels 9.0% / 9.7%. The sign held on 4/4 |
| P5 | B1 never changes the top-1 model; B3 flips a significant pair | **pass** |
| P6 | The no-refit forecast misses alteration bugs | **pass** |

**Deviation D2: E2 is degenerate on the leverage panels.**
- One BHC holds 68% of the weight, with leverage up to 14,699.
- Pooled ρ is 0.12, against a per-bank median of 0.87.
- Winsorizing (x7, a "bug" here) moves ρ by 0.71–0.79. Here it is arguably a fix.
- E2 is graded on the cash panels only (top-10 weight share 4.5%).
- Lesson for the paper: an audit is relative to the estimand, and a reference run can itself be fragile. The decomposition exposes this, because a weight-space TV of 0.87 from dropping a single bank is a red flag.

**What changed in the picture:**
1. **Coverage vs bias is stable across full n and across estimands** (cash panels):

   | Bug | E1 shift | E2 shift | Class |
   |---|---|---|---|
   | B1 | 0.8–2.6% | 0.0–4.1% | coverage |
   | B2 / x3 | 16–24% | 17–22% | bias |
   | x4 | 4–26% | 9–10% | borderline / bias, depends on the panel |

2. **Rankings are robust to every selection bug.** Across 4 panels × 4 selection bugs, no significant pair flipped and the top-1 model never changed. Every ranking flip comes from alteration bugs (B3, B5 scale leak, x5, x7), which the ledger cannot see by design. For model-comparison papers, alteration errors are the bigger threat. That is the main limitation and should be stated up front.
3. **Harm under the CI rule depends on n; materiality does not.** B1's harm on bank_leverage went from 0.57 (4k subsample) to 0.90 (full panel), while its relative shift stayed under 3%. The paper should classify errors by materiality and report CI harm as secondary.

## 3f. LLM-written pipelines (`llm_bugs/`, frozen before any real-data run)

**Setup.**
- Three models (Haiku, Sonnet, Opus) × 5 runs each wrote `build_panel(calls, attrs)`. Each run got an identical prompt and its own folder containing only the spec and a synthetic sample.
- Every pipeline ran on all four full panels (inputs: real values plus a synthetic, outcome-independent charter table).
- Comparison: against the intended reference (`llm_bugs/reference.py`), using ledger v1 and the frozen decomposition v2.

**Arm A, explicit spec (every convention written down):** 15/15 pipelines reproduce the reference **exactly** in 60/60 cells. The ledger raised 0 false flags. With every convention stated, current LLMs make no convention errors on this task. Q1 and Q6 failed; everything else was vacuous.

**Arm B, under-specified spec** ("put it on a regular quarterly frequency, handle gaps sensibly, leave out banks with poor coverage"):

| | Prediction | Result |
|---|---|---|
| QB1 | ≥ 8/15 diverge | **pass.** 15/15 diverge, by 0.6–7.4% of entity-periods; all pass schema and stats checks |
| QB2 | ≥ 3/15 impute | **pass.** 15/15 linearly interpolate gaps of 2–4 quarters |
| QB3 | ≥ 3/15 assign charter from a single snapshot | **fail.** 0/15; all use point-in-time charter |
| QB4 | Ledger flags divergences | **pass.** 60/60 |
| QB5 | Some pipeline is E1-material on ≥ 2 panels | **fail.** Only 2/60 cells (Sonnet 4 = Sonnet 5, bhc_cash, 10.6%) |
| QB6 | T3 = 0 when there are no value mismatches | **fail as written.** Interpolation alters series through *added* keys |

**What this shows:**
1. **Spec explicitness is the switch.** Explicit spec: 0/15 diverge. Implicit spec: 15/15 diverge. LLM convention errors come from unstated conventions, not from coding mistakes. This matches how the motivating grid bug arose (an unstated step convention).
2. **The models converge on one set of conventions.** All 15 pick variants of the same recipe:
   - interpolate gaps of ≤ 2–4 quarters;
   - require 75–80% coverage and a minimum length of 12–24 quarters;
   - often keep only the most recent contiguous segment.

   Two runs wrote functionally identical code. The E1 shift has the **same sign in 54/60 cells**, always favouring Naive. So averaging over several LLMs (a "many analysts" design) would not cancel the error.
3. **The shifts are mostly coverage-type, near the threshold.** E1 shifts are 0–4.5% on the bank panels and up to 10.6% on bhc_cash. E3 is never changed. About 74% of each shift comes from selection (median); the rest is interpolation (T3).
4. **The ledger cannot triage on its own.** It flags 60/60 cells, because interpolated keys break provenance. The decomposition is what separates the pure-interpolation runs (Haiku 2–4: TV ≈ 0, shift ≤ 0.6%) from the segment-and-length runs (TV up to 0.035, shift up to 10.6%). That is the intended division of labour, and the paper should present ledger and decomposition as one tool.

## 3g. Second task and second model family (frozen before generation)

**Setup.**
- Task 2 is a year-over-year change panel (`spec_A2.md` explicit, `spec_B2.md` under-specified), checked against `reference_yoy.py`.
- The second model family is OpenAI `gpt-6.2-sol`, run through the `codex` CLI bundled with ChatGPT.app at its default low reasoning effort. It saw only the spec and the synthetic sample.
- Pipelines:
  - A2: 4 families × 2;
  - B2: 4 families × 5;
  - Codex also ran task-1 arm B × 5.
- Results: `llm_bugs/{A2,B2,Bcodex}_results_*.csv`. Grading is in `FREEZE_LEDGER.txt`.

**Score: 2 of 9 predictions pass** (T2A, R1). R6 is vacuous. The failures are informative:

| | Result |
|---|---|
| **Explicit spec (T2A)** | 8/8 exact. Across both tasks and all 4 families, **23/23 explicit-spec pipelines are exact on all 92 panel cells.** |
| **Under-specified (R1)** | 20/20 diverge on task 2. Including task 1, **40/40 under-specified pipelines diverge**, and all pass schema checks. |
| **Expected classic bugs (R2, R3)** | **Failed.** 0/20 lag by rows and 0/20 use a percent change. Current models avoid the textbook traps; two Opus runs explicitly reject percent change because the ratio can be near zero. |
| **Materiality (R4)** | **Failed.** Task-2 divergences move E1 by ≤ 3.9% and never change the ranking. Many extra rows belong to banks with 8–15 year-over-year quarters, which the evaluator's own 16-quarter minimum drops, so E1 is exactly unchanged. |
| **Correlation (R5)** | **Failed narrowly.** Same-sign share is 0.71 on task 2, and 0.775 on task 1 across all four families. |
| **Codex (C1)** | **Failed.** Codex diverges 5/5 on task 1 but **forward-fills** (4/5) where the Anthropic models interpolate. |
| **Cross-family agreement (C2)** | **Failed.** On task 1, Codex's median shift has the opposite sign on both BHC panels and is about 10× smaller. On task 2, the Anthropic median is exactly 0 while Codex is negative. |

**Revised picture (this replaces the "shared default" claim in 3f):**
1. **Explicit specs remove the errors entirely; under-specification produces them every time.** This holds across 2 tasks and 4 families: 23/23 exact vs 40/40 divergent.
2. **Conventions converge within a family but differ across families.**
   - On task 2, the 15 Anthropic pipelines produce only 4 distinct outputs; all 5 Opus runs are identical.
   - Codex produces 4 distinct outputs, none shared with Anthropic.
   - On task 1, the families split on imputation (interpolate vs forward-fill) and on the direction of the shift.
   - So errors are correlated *within* a model family. Mixing families diversifies conventions, though it does not make any single one correct.
3. **Severity depends on the task.**
   - Task-1 conventions (keep the latest segment, interpolate) are non-uniform in life-cycle terms. They shift E1 by up to 10.6% and the ledger flags them (80/80).
   - Task-2 conventions (a minimum-length threshold, checking charter at one end only) are near-uniform. They shift E1 by ≤ 3.9% and the ledger stays silent (0/80).
   - In both tasks, ledger silence lines up with immateriality and ledger flags line up with the larger shifts. That is the best evidence so far that the ledger tracks what matters, though it still over-flags relative to the 10% materiality bar.

## 3h. Third task: a planted survivorship trap (frozen before generation)

**Setup.**
- The task is a Midwest-bank panel. The inputs include a `directory` "as published at the end of the sample period". It lists only surviving banks: 22–28% of all banks on the real panels.
- A full headquarters `history` is also provided. The correct answer uses it point-in-time.
- The A3 spec says not to use the directory; B3 describes it neutrally.
- Pipelines: 4 families × (2 A3 + 5 B3) = 28, each run on 4 full panels.
- Results: `llm_bugs/A3_results_*`, `B3_graded.csv`.

**Result: the trap caught nothing.**
- S1 failed: **0/20** B3 pipelines filtered on the directory. The largest exited-bank loss was 11.8%, and it comes from gap handling. All 20 took state point-in-time from the history.
- Haiku, Opus and Sonnet hand-backs explicitly name the survivorship or look-ahead risk, including Haiku.
- A3 was exact in 8/8.
- One Codex run crashed on every panel (loud, not silent). It filled missing end dates with `inf`, which works on the CSV sample (float) but fails on the production parquet (nullable Int64). This is logged as deviation D4, a sample/production dtype mismatch.

**What did shift results:** the same generic conventions as in task 1.
- Keeping only the most recent contiguous stretch with a ≥ 24-quarter minimum selects on history length, which is outcome-correlated. That makes it a bias-type selection.
- It produced the only material cells: 3/76, Opus 2, Opus 3 and Sonnet 5 on bhc_cash, at 10.0–10.4%.
- Selection accounts for 86% of each shift (median). The ledger flags all 76 cells.

**Implication for the paper.** Current LLMs reliably avoid *recognizable* bias sources, the textbook survivorship join, even at low reasoning effort. The residual risk is in *unrecognized* conventions: gap and segment rules that look like routine cleaning but select on life-cycle coordinates. The planted-trap design cannot show "LLMs fall for survivorship"; it shows the opposite. The paper's message becomes: *the dangerous conventions are the ones nobody names, and they are exactly what the ledger plus decomposition measures.*

## 3i. Ablation of the LLM conventions (`llm_bugs/ablation.py`, frozen before the run)

**Setup.**
- Starting point: the task-1 reference.
- Factors, switched on one at a time and in combination:
  - G: interpolate gaps of ≤ 2 quarters;
  - S: keep the most recent contiguous segment;
  - L: minimum length of 16, 20 or 24 quarters.
- Design: 2×2×4 = 16 configs, each run on 4 full panels.
- Results: `llm_bugs/ABL_results_*`; grader: `grade_abl.py`.

**Score: 4 of 7 predictions pass.** The failures overturn my own mechanism story:

| | Result |
|---|---|
| AB0 | **Pass.** G0_S0_L0 equals the reference on all panels. |
| AB1 | **Pass.** Interpolation alone moves E1 by ≤ 0.5%. |
| AB2 | **Fail.** The **minimum length alone is the driver.** L24 shifts E1 by 1.2–10.4%. L16 shifts nothing, because the evaluator already requires 16. |
| AB3 | **Fail.** The segment rule is negligible (≤ 0.5%, mixed sign). I had predicted it was the main driver. |
| AB4 | **Pass.** G+S+L24 reproduces the LLM pipeline's shift at 0.98–1.06× on every panel, so the ablation explains the LLM result. |
| AB5 | **Pass.** The shift is ≥ 98% selection. |
| AB6 | **Fail.** The segment rule alone is *not* flagged, being near-uniform over the life cycle. L16 *is* flagged on the BHC panels while being harmless. |

**Exploratory dose-response (not pre-registered): minimum history length L vs E1 shift.**

| L | bank_cash | bank_lev | bhc_cash | bhc_lev |
|---|---|---|---|---|
| 20 | 2.3% | 0.6% | 5.3% | 3.6% |
| 24 | 4.7% | 1.2% | 10.4% | 6.7% |
| 32 | 9.5% | 2.4% | 18.7% | 11.8% |
| 48 | 22% | 6.9% | 35% | 24% |

The shift is monotone in L. At L24 only 6% of bhc bank-quarters are dropped, yet the headline moves 10%. Every step is in the same direction: Naive looks better, because short-history banks, mostly entrants, favour HistoricAverage. The top-ranked model never changes.

**Takeaways for the paper.**
1. **One unstated number does the damage.** Every material LLM divergence in tasks 1 and 3 traces to the minimum history length. LLMs derive it as "N training quarters + 4 held out" and set it above the evaluator's own 16.
2. **The harm is a dose-response.** It is a smooth function of a single convention parameter, not a discrete bug. A paper figure: E1 shift vs L, with each LLM's chosen L marked on the curve.
3. **This is selection on history length**, the same mechanism as held-out bug x4. The decomposition attributes it fully to selection.
4. **The ledger's sensitivity is imperfect at small scale.** It misses the near-uniform segment rule, which is harmless, and over-flags L16, also harmless. It catches L ≥ 20. This argues for reporting the decomposition's materiality next to the flag rather than the flag alone.

## 3j. Migration to hanqing-win and non-bank replication (EIA-923 power plants)

**Where things run now.** Following `Research_Porfolio_2026/RULES.md`, P18 runs on hanqing-win:
- Location: `Research_Portfolio_2026\p18-convention-errors`, with its own `.venv`.
  - Python 3.12.10, pinned in `requirements.txt` and `requirements.lock.txt`.
- One entry point: `scripts/run_all.py`.
  - Stages: `pilot`, `realdata`, `estimands`, `llm`.
  - Environment variables `P18_PANELS`, `P18_OUT`, `P18_WORK` select the panel set and the output and work folders.

**Reproduction.** All earlier results (from the Mac scratchpad) were re-run on Windows and match file for file:
- numeric differences ≤ 2e-10 relative;
- categorical results identical.

Details are in `FREEZE_LEDGER.txt` and `scripts/compare_runs.py`. Deviation D5: the runner passes forward-slash paths because the frozen scripts take panel tags with `split("/")`.

**EIA panel.**
- Source: 24 annual EIA-923 files, 2001–2024, 331 MB, listed in `data/raw/eia923/MANIFEST.csv` on Windows.
- Builder: `scripts/build_eia.py`.
- Content: quarterly net generation for 15,131 plants, 686,908 plant-quarters.
  - Entry and exit are the first and last non-zero quarter.
  - 4,387 plants are present at the start; 12,517 are still alive at the end.
- Five of the six predictions were confirmed (N1, N3–N6), and N2 failed. Results are in `results_win_eia/`.

| | Prediction | Result |
|---|---|---|
| N1 | Ledger v1 out of sample | **pass.** 8/8 flagged, 0/6 false |
| N2 | P12 grid bug B1 is coverage-only | **FAIL.** On power plants B1 is material: **17.3% shift**, harm ratio 1.7, all selection. Survivorship bugs B2/x3: 48% / 42% |
| N3 | No-refit decomposition exact | **pass** |
| N4 | Minimum-length dose-response | **pass,** and much steeper: 9.6% / 20% / 45% / 82% at L = 20 / 24 / 32 / 48 |
| N5 | LLM explicit exact, under-specified divergent | **pass.** 15/15 exact; 20/20 diverge; ledger flags 20/20 |
| N6 | Material LLM cells all come from min length > 16 | **pass.** 7/7 (min length 20 or 24) |

**The N2 failure is the key result.** In the bank panels, the quarter an entrant first reports carries no outcome signal, so the grid bug costs coverage, not accuracy. In power plants it does carry signal. The mean forecast gap is +0.02 for Q3–Q4 entrants, −0.13 for Q1–Q2 entrants and −0.57 for incumbents (exploratory, `scripts/eia_entry_q.py`). The mechanism, technology mix or seasonal generation, is untested.

So **the same line of code is a coverage error in one domain and a bias error in another.** Whether a dropped coordinate is outcome-irrelevant is an empirical property of the panel, not of the bug. That is exactly what the no-refit decomposition measures.

This turns P12's story into a general claim: one cannot certify a convention bug as harmless by inspecting the code; one must measure it against the panel.

**Mechanism (EIA-860 2024 technology, `scripts/mechanism_eia.py`; frozen predictions M1–M5).**
- **Score:** 3 of 5 pass; the 2 failures are narrow.
- **Data:** 93% of plants are matched to a technology, taken from their largest generator.
- **The mix differs (M1 pass).** Technology mix differs sharply by entry quarter (p ≈ 1e-187).
  - The solar share of entrants is 44% in Q1 and 62–63% in Q3–Q4.
  - Petroleum is 11% of Q1 entrants and ≤ 3% later.
- **Technology is the largest channel, but only a partial one.**
  - **M2 fail:** technology fixed effects cut the entry-quarter spread in the outcome by 40% (0.153 → 0.092), against a predicted ≥ 50%.
  - **M3 pass:** history length and entry year explain less (spread 0.119–0.137).
  - **M4 fail:** with technology and entry-year controls, the leftover entry-quarter effect has p = 0.049, just under the predicted > 0.05.
  - **M5 pass:** decomposing the grid bug's shift by technology cells puts 64% of its selection effect between technologies.
- **Reading.** The bug is harmful on plants mainly because it drops a technology-skewed slice: Q2/Q3 entrants, which are solar-heavy. A residual within-technology entry-quarter effect remains and is noisy in small cells.
- **Caveat.** The mechanism is partly identified, not fully. The paper should say "technology composition accounts for most of the selection effect," not "the effect is explained."

## 3k. Propositions (`paper/PROOFS.md`)

Four propositions are proved, and each is checked numerically on all 5 panels (outputs in `results_win/proofs/`):
1. **Exact identity and no-refit forecast** for weighted-mean estimands, which also explains formally why alterations are invisible to the ledger.
2. **Composition bounds:** |T1| ≤ TV·range(μ) and |T1| ≤ √χ²·σ_B. Composition change is harmless when μ is flat along the selected coordinate.
3. **Within-cell bound for T2:** T2 = 0 when selection is constant within cells.
4. **Minimum-length shift:** shift = (1−π_L)(μ_{≥L} − μ_{<L}), with a monotonicity condition. It matches the 50 ablation results to 8e-16.

The monotonicity premise of 4(b) holds only approximately in the data.

## 3l. Paper draft (`paper/main.tex`, 15 pages)

- **Build the numbers, figure data and tables:** run `scripts/make_paper_values.py` **on hanqing-win** with the project `.venv`, then pull `paper/generated/` back.
  - It writes 66 values, two figure data files and five tables.
  - Every data value in the text is a `\val{}`, so nothing is hand-typed.
- **Compile:** on alex-macbook (`/Library/TeX/texbin/pdflatex`, run three times). This Mac has no LaTeX.
- **Figures** are pgfplots drawn from the generated `.dat` files:
  - Figure 1, composition shift (TV) vs relative shift;
  - Figure 2, the minimum-length dose-response with the LLM thresholds marked.
- **Appendices:**
  - A, proofs;
  - B, the pipeline catalogue;
  - C, all 56 pre-registered predictions with their outcomes (25 pass, 23 fail, 8 vacuous).
- **Open TODO:** verified citations for the agentic data-science papers in Related work.

## 4. Next steps (about 7–8 weeks)

1. **Week 1.** Unblock the LLM reviewer. Run 2+ models × 6 pipelines × 5 reps, plus a "with ledger output" arm.
2. **Weeks 1–2.** Build the held-out bug set:
   - Have LLM agents write the panel pipeline from a spec on real data (P12 Call Reports, plus one non-finance panel). Harvest their natural bugs.
   - Mine 20–40 real calendar/merge bugs from GitHub issues (pandas `DateOffset`, `resample` label/closed, `merge_asof` direction, fiscal-quarter `Q-JUN` periods).
3. **Weeks 3–4.** Freeze the protocol, then run the ledger, baseline checks and LLM reviewers on the held-out set. Measure the conclusion shift on real data.
4. **Weeks 5–6.** Write up a theory note: when non-uniform retention implies bias (selection on life-cycle coordinates correlated with the outcome). Release the ledger as a ~100-line package.
5. **Weeks 7–8.** Write the paper. Possible venues: a NeurIPS / ICML data-centric or evaluation workshop, or JDS / Stat.

**Compute.** The full study needs no GPU. If it uses the company server, follow `option-strategy/RULES.md`: write only under `/mnt/data-hdd/a100_data/yanghanqing`. The AISC service is shared backtest infrastructure, not a general LLM endpoint, so do not use it for this.
