# Paper outline (draft v0, 2026-11-19)

> **Superseded for writing by `paper/main.tex` (draft v1, 2026-11-20).** Numbers there come from `scripts/make_paper_values.py` -> `paper/generated/`. This file remains the planning record.

## Working title

**Coverage or Bias? Auditing Silent Composition Changes in Panel-Data Pipelines Without Refitting**

Alternative titles:
- "Most silent panel bugs shrink your population, a few bias your answer, and standard checks can't tell which"
- "Who Is Still in the Sample? An Entity-Period Ledger for Longitudinal ML Pipelines"

## One-sentence claim

Silent preprocessing bugs in longitudinal panels mostly change *which entities a conclusion covers* rather than *the conclusion itself*. Standard checks see neither. A raw-vs-processed entity-period ledger plus an exact decomposition separates coverage errors from bias errors, and it does so using only the reference run, with no refitting.

## Abstract (draft, ~180 words; numbers are pilot numbers and will change)

Data pipelines for panel data can silently change sample composition. They run without error, pass schema and distribution checks, and yield plausible results. A recent example is a public financial-forecasting benchmark: its quarterly grid convention excludes about 35% of bank entrants, yet the change went unnoticed.

We study this failure class on four U.S. Call Report panels (1976–2020) and a synthetic panel with known structure. We use 14 pipeline variants, including 9 bug patterns taken from documented pitfalls.

We propose (i) an entity-period ledger that measures retention along entity life-cycle coordinates (entry cohort, age, exit) relative to a declared reference pipeline. We also propose (ii) an exact three-term decomposition of a pipeline's effect on the headline estimate: between-cohort composition, within-cohort selection, and series alteration.

We pre-registered both. The ledger detected all 8 composition bugs on a held-out panel, with no false alarms on 6 non-composition or benign variants. For selection-only bugs, the decomposition recovers the full effect from the reference run alone. Composition magnitude does not predict harm: a calendar-convention bug moving 41% of cohort mass shifted the estimate by under 0.6 CI half-widths. Survivorship and history-length filters moving 10–70% shifted it by 1.1–5.5 half-widths.

## 1. Introduction

1. **Motivation.** ML and statistics on longitudinal panels (finance, health, education, ops) rely on preprocessing conventions: calendars, frequencies, joins, filters. Every operation can be locally correct while the analyzed population silently changes.
2. **Motivating incident.** In the P12 FTSFR case, the `QE -> 3mo` step lands 30 Jun / 30 Sep entrants on 30 Dec. About 35% of post-1976 entrants are dropped, and the benchmark count matches the buggy arm (17,383).
   - It survived an extensive audit pipeline.
   - Ranking conclusions differ between entrants and incumbents.
3. **Gap.**
   - Agentic and DS verification work (RADAR, sanity checks, workflow harnesses, ds-agent-harness) checks operations or final answers.
   - Leakage work (Kapoor & Narayanan; notebook static analysis) targets train/test contamination.
   - Neither asks *who is still in the sample*, or whether a composition change matters.
4. **Contributions.**
   - C1. A taxonomy distinguishing **coverage errors** from **bias errors** among silent composition changes, with the alteration class kept separate.
   - C2. An **entity-period ledger**: differential against a reference pipeline, life-cycle coordinates, key-only provenance, retention floor.
   - C3. An **exact decomposition** (T1/T2/T3) showing that selection-only effects are computable without refitting.
   - C4. **Pre-registered evidence** on 4 real panels plus synthetic data, including a failed prediction and a failed ledger version. These are reported, not hidden.
   - C5. A re-reading of the motivating benchmark: its error is a coverage error, not a bias error.

## 2. Related work

- **Agentic and automated data science reliability:**
  - RADAR / "Fail Loudly" (2609.32528)
  - Sanity Checks for Agentic DS (2604.11003)
  - Workflow harnesses (2609.35255)
  - When Tools Silently Lie (2609.37153)
  - ds-agent-harness
  - Benchmarks: DSBench, MLE-bench
- **Analytic flexibility:** Many AI Analysts (2602.18710), Agentic Garden of Forking Paths (2607.01507), multiverse analysis.
- **Leakage:** Kapoor & Narayanan taxonomy; notebook leakage static analysis (2209.03345); the 2,047-dataset leakage landscape (2604.04199).
- **Data validation:** schema/expectation testing (Great Expectations-style, TFDV), distribution-shift tests. These are our baselines.
- **Survey and econometrics:** survivorship bias (finance: CRSP delisting, Compustat back-fill); Kitagawa / Oaxaca–Blinder decompositions (our T1/T2 is their pipeline analogue); selection on observables vs MAR.
- **Benchmark construction:** FTSFR, and reproducibility of financial ML benchmarks.

## 3. Problem setup

- A panel is a set of (entity, period, value) records. A pipeline maps raw records to processed records. An estimand is a headline statistic computed on the processed panel.
- Life-cycle coordinates are computed from raw data only: entry cohort (incumbent vs entrant × entry calendar quarter), age since entry, and exit status.
- **Definitions:**
  - *composition change*: non-uniform retention along those coordinates relative to a reference;
  - *coverage error*: a composition change whose effect on the estimand is within reference uncertainty;
  - *bias error*: a composition change whose effect exceeds it;
  - *alteration error*: retained keys with changed values (ffill, stale links, leakage).
- Full statements and proofs: `paper/PROOFS.md`. Each is checked numerically on 5 panels (`results_win/proofs/`).
- **Proposition 1 (identity, no-refit).** ΔΔ = T1 + T2 + T3 for any weighted-mean estimand. For selection-only pipelines T3 = 0, so the shift and the full ranking are computable from reference per-entity results plus the retained set. This also explains, formally, the blind spot for alterations.
- **Proposition 2 (composition bound).** |T1| ≤ TV·range(μ_r) (sharp) and |T1| ≤ √χ²·σ_B. Composition change is necessary but not sufficient; heterogeneity along the selected coordinate is required. If μ is constant across the reweighted cells, T1 = 0 (coverage only).
- **Proposition 3 (within-cell term).** μ_kc − μ_rc = Cov_c(a, k)/π_c, so |T2| ≤ Σ W_kc σ_rc √((1−π_c)/π_c). T2 = 0 when selection is constant within cells.
- **Proposition 4 (minimum length).** Shift = (1−π_L)(μ_{≥L} − μ_{<L}); zero at L ≤ evaluator minimum; monotone in L if E[a | n] is monotone in n. Matches the ablation to 8e-16.

## 4. Method

### 4.1 Entity-period ledger (v1)

Rules R1–R3, as frozen (`pilot/ledger_v1.py`, hash `ba7ac1a6…`):
- R1 key provenance;
- R2 retention floor (0.5× reference);
- R3 differential retention spread ≤ 0.10 across coordinates.

Cost: one pass over keys, with no model.

### 4.2 Decomposition

Algorithm and complexity. Shows which inputs are needed: the reference per-entity results and the candidate's retained set.

### 4.3 Design history

v0 compared against raw data and failed on real data (100% false alarms), because legitimate quality filters are non-uniform. This motivates the differential design. Report it in the main text; it is a lesson for practitioners.

## 5. Experimental design

- **Panels:**
  - FTSFR-style NYU Call Report panels: bank_leverage, bhc_leverage, bhc_cash_liquidity, bank_cash_liquidity, 1976Q1–2020Q1, 4,000-entity subsamples (TODO: full panels).
  - A synthetic panel with known incumbent/entrant dynamics.
- **Estimand:** mean per-entity MASE(Naive) − MASE(HistoricAverage), h = 4.
  - TODO: a second estimand family (cross-sectional regression coefficient) and a ranking estimand over the six P12 models.
- **Pipelines:**
  - Design set B1–B5, written while the ledger was designed.
  - Held-out set x1–x9, written after the ledger freeze from documented pitfalls: pandas offset anchoring, current-constituent survivorship, CCM many-to-many links, `shift(-1)` + `dropna`, winsorize, z-score, and benign controls.
  - TODO: a third set from LLM-written pipelines and mined GitHub issues.
- **Baselines:**
  - schema checks;
  - distribution checks (row retention, mean shift, KS);
  - TODO: an LLM code reviewer given code plus run summary, with and without ledger output.
- **Ground truth:** "harmful" means the estimate leaves the reference 95% CI. This is defined without our judgment.
- **Pre-registration:** `FREEZE_LEDGER.txt` holds hashes plus predictions with timestamps. Report every prediction and its outcome.

## 6. Results

### 6.1 Silent harmful bugs exist on real panels

Table 1: the pipeline × panel matrix of shift, harm, schema, stats and ledger v1.

- Schema checks never fire.
- Distribution checks miss x4 (−16%, bhc_lev) and B4 (+6%, bank_lev).

### 6.2 The ledger detects composition changes out of sample

The pre-registered test on bhc_leverage: 8/8 flags, 0/6 false alarms.

Boundary: no detection of alteration bugs (B5, x5), by design.

### 6.3 Coverage vs bias (headline result)

**Figure 1:** cohort TV (x) vs harm ratio (y), 4 panels × 5 composition bugs.

Full panels. Shift is relative (n-free materiality); E2 is reported on the cash panels only (see 6.5).

| Bug | Selects on | TV (E1) | E1 shift | E2 shift | Ranking | Class |
|---|---|---|---|---|---|---|
| B1 (motivating bug) | entry calendar quarter | 0.15–0.41 | 0.8–2.6% | 0.0–4.1% | unchanged | coverage |
| x4 | history length | 0.09–0.11 | 4–26% | 9–10% | unchanged | bias (panel-dependent) |
| B2 / x3 | exit status | 0.64–0.76 | 16–24% | 17–22% | unchanged | bias |

Headline: materiality, not CI harm, defines the class. CI harm depends on n: B1 went from 0.57 to 0.90 between the subsample and the full panel.

### 6.3b Rankings are robust to selection, fragile to alteration

- Across 4 panels × 4 selection bugs, no significant pairwise flip occurred and the top-1 model never changed.
- Every ranking flip (drift vs SES, SNaive vs drift) comes from alteration bugs: B3 ffill, B5 scale leak, x5 target shift, x7 winsorize. The ledger is blind to these by design.
- This gives a scope statement: the ledger plus decomposition audits *level* estimands. *Comparative* estimands need alteration checks.

### 6.4 The decomposition is exact and needs no refit

- Pre-registered D2 passed: T3 share 0.00 for selection bugs and ≥ 1.0 for alteration bugs.
- Figure 2: stacked T1/T2/T3 bars per bug.
- A practical demo: how long the ledger plus decomposition takes vs refitting six models.

### 6.5 Failed predictions (reported)

- D1 and D3 predicted that B1 would be harmful on the cash panels. Both failed.
- Why: the selected coordinate (entry calendar quarter) is not associated with the outcome. This is consistent with the motivating case's own small MASE effect (0.2–1.3%).
- Lesson: composition magnitude is a poor proxy for harm.
- Full-panel round (P1–P6): 4 of 6 passed.
  - P2 failed: B1 harm was 0.90, not > 1.
  - P4 failed: x4's E2 shift was 9.0% / 9.7% against a 10% threshold.
- Deviation D2: the persistence estimand is degenerate on the leverage panels (one BHC carries 68% of the weight).
  - We report this as a finding: a fragile *reference* estimand shows up as a huge weight-space TV when a single entity is dropped.
  - Here winsorizing, nominally a "bug", is arguably the fix.

### 6.5b Where convention errors come from: LLM-written pipelines

**Design.**
- 2 tasks: (1) a level panel with charter filtering and gap handling; (2) a year-over-year change panel.
- 4 model families: Haiku, Sonnet, Opus, and OpenAI gpt-6.2-sol via `codex`.
- 2 arms per task: an explicit spec and an under-specified spec.
- 63 pipelines in all, each run on 4 full panels.

**Table: divergence and materiality by task × arm × family.**
- *Explicit spec:* 23/23 pipelines exact (92/92 cells).
- *Under-specified, task 1:* 20/20 diverge. The ledger flags 80/80. E1 shifts reach 10.6% (2 material cells).
- *Under-specified, task 2:* 20/20 diverge. The ledger flags 0/80. E1 shifts are ≤ 3.9% (0 material).

**Claims for the paper (revised after the second family; the shared-default claim was too strong):**
1. Convention errors are *specification* errors, not coding errors. The classic coding traps did not occur: 0/20 lagged by rows, 0/20 used a percent change.
2. Conventions converge within a model family and differ across families.
   - Task 2: the 15 Anthropic pipelines produce 4 distinct outputs, and all 5 Opus runs are identical.
   - Codex produces 4 distinct outputs, none shared with Anthropic.
   - Task 1: the families split on interpolation vs forward-fill, and the shift reverses sign on the BHC panels.
3. Severity depends on the task. Conventions that are non-uniform over the life cycle (keeping the latest segment) are flagged and can be material. Near-uniform threshold choices are neither, and some are absorbed by the evaluator's own filters.
4. Ledger plus decomposition triages these divergences. The ledger's silence lines up with immateriality, but its flags still over-call relative to the 10% bar.
5. **A planted survivorship trap (task 3) caught 0/20 pipelines in all 4 families.** Several runs named the risk explicitly. The only material shifts (3/76 cells) came from the unnamed "latest contiguous segment + minimum length" convention, which selects on history length. LLMs avoid named biases; the residual risk is in conventions that look like routine cleaning.
6. **Ablation: one number does the damage.** A 16-config factorial replicates the LLM shift at 0.98–1.06×. The minimum history length alone drives it: L24 shifts E1 by up to 10.4%. The segment rule and gap interpolation contribute ≤ 0.5%. The effect is a monotone dose-response in L, from 0 at the evaluator's 16 to 7–35% at 48.
   - Planned figure: E1 shift vs L, with each LLM's chosen L marked.
7. **Non-bank replication (EIA-923 power plants, 15,131 plants): the same bug flips class.** The P12 grid bug is a coverage error on all four bank panels (≤ 2.6%) but a *bias* error on power plants (17.3%, entirely selection).
   - Mechanism: entry quarter is outcome-relevant for plants. The mean forecast gap is +0.02 for Q3–Q4 entrants vs −0.13 for Q1–Q2 entrants; for banks, entry quarters are indistinguishable.
   - Also replicated on EIA, pre-registered: ledger 8/8 with 0/6 false; exact decomposition; minimum-length dose-response (82% at L48); LLM explicit 15/15 exact; every material LLM cell has min length > 16.
   - Headline candidate: *harmlessness is a property of the panel, not the code.*
   - Mechanism, pre-registered (M1–M5, 3 of 5 pass): entry quarter carries technology mix. Solar is 44% of Q1 entrants and 63% of Q4 entrants (p ≈ 1e-187). Technology fixed effects remove 40% of the entry-quarter outcome gap and 64% of the bug's selection effect; history length explains less. A residual remains (p = 0.049), so the claim is "mostly technology composition", not full identification.

This section probably becomes the paper's second headline, next to coverage vs bias.

### 6.6 (TODO) LLM reviewers

Detection rates by error class, with and without ledger output.

Hypothesis: reviewers notice survivorship joins, which are textbook, more than calendar or offset bugs.

## 7. Case study: the motivating benchmark

Re-read the P12 finding through the taxonomy:
- The grid bug is a coverage error for aggregate accuracy.
- It removes a third of entrants, so any claim about entrants rests on the retained Q1/Q4 entrants.
- Exchangeability holds empirically: excluded entrants behave like retained ones.
- Recommendation for benchmark maintainers: publish a ledger table.

## 8. Discussion and limitations

- **Harm is relative to an estimand.** Heterogeneous-effect, subgroup or ranking estimands can turn coverage errors into bias errors. The case study's entrant/incumbent ranking reversal is exactly that.
- **The reference pipeline is a choice.** The ledger audits *differences* between pipelines, not absolute truth.
- **Coordinates are domain-specific.** Life-cycle coordinates fit entry/exit panels (firms, patients, users). Other domains need their own.
- **Scope.** Value-level and leakage errors are out of scope; pair the ledger with existing leakage tools.
- **Data scope.** U.S. bank panels and one estimand family so far. Subsamples of 4,000 entities.
- **Implications for agentic DS.** Agents that write pipelines should emit ledgers. An agent's verifier should check composition, not just code.

## 9. Conclusion

## Appendix plan

- A. Bug catalogue: code, source of each pattern, and the expected mechanism.
- B. Pre-registration log: the full `FREEZE_LEDGER.txt` and its deviations (D1: real-data B1 horizon).
- C. Synthetic DGP and 3-seed results.
- D. Ledger v0 results (the failure).
- E. The bank-level cash panel construction.

---

## Evidence map

| Claim | Evidence in hand | Still needed |
|---|---|---|
| Silent harmful bugs exist on real data | estimands_*_full.csv (4 full panels) | — |
| Ledger detects out of sample | bhc_leverage pre-reg pass; v1 flags hold on 4 full panels; LLM arm B 60/60 | GitHub-mined bugs; a non-bank panel |
| LLM convention errors come from under-specification; correlated within family | 2 tasks × 4 families: explicit 23/23 exact, under-specified 40/40 diverge; within-family collapse, cross-family split | Done: task 3 trap; ablation; EIA replication; EIA-860 mechanism (technology ~40–64%). Next: proofs (Prop. 1–2); second non-Anthropic family; residual within-tech effect (capacity size, commissioning month) |
| Coverage vs bias; TV ≠ harm | 4 full panels × E1/E2/E3 (P3 pass) | More bugs per coordinate; a non-bank panel; a regression estimand that is robust to outliers on leverage |
| Decomposition exact, no refit | D2 pass; P1 pass on 3 estimands; Props 1–4 proved and checked on 5 panels | Timing benchmark |
| Rankings robust to selection, not alteration | P5 + 6.3b | An alteration-side check to pair with the ledger |
| LLM reviewers miss calendar bugs | — | Whole experiment (CLI login) |
| Case-study re-reading | P12 confirm results | Ranking estimand on the P12 full run |

## Remaining work (~6 weeks)

1. **Week 1.**
   - Run full panels, not subsamples.
   - Add the regression estimand (e.g. cash ratio on size, with entry-cohort FE) and the 6-model ranking estimand.
   - Freeze predictions first.
2. **Week 2.** Build the third bug set:
   - 3 LLMs write the pipeline from a spec, 5 runs each; harvest their bugs.
   - Collect 20 bugs mined from GitHub / StackOverflow.
   - Freeze the ledger v1 test before running.
3. **Week 3.** Run the LLM-reviewer experiment (needs `claude /login` or an API key).
4. **Week 4.** Write the proofs (Prop. 1–2) and a timing benchmark. Package the ledger (~150 lines, pandas/polars).
5. **Weeks 5–6.** Writing, figures, appendix.

**Venue options:**
- NeurIPS 2027 Datasets & Benchmarks or an evaluation workshop
- the ICML data-centric workshop
- *Journal of Data Science* / *Stat*
- *Journal of Financial Data Science*, if the case study is emphasized
