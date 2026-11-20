# Propositions and proofs (P18, draft v1, 2026-11-19)

Notation follows `pilot/estimands.py` (decomposition v2) and the outline (section 3). Every result here is elementary. The point is to state precisely *what* the no-refit audit can and cannot certify, and to give the conditions under which a composition change is harmless.

## Setup

**Entities.** $U$ is the set of raw entities, and $c: U \to \mathcal C$ assigns each entity a **cell**, computed from raw data only (for example entry cohort × exit status). Cells partition $U$.

**Estimands as weighted means.** A pipeline $q$ (reference $r$ or candidate $p$) together with the evaluator yields:
- an evaluated set $S_q \subseteq U$;
- for each $i \in S_q$, a contribution $a^q_i \in \mathbb R$ and a weight $w^q_i > 0$.

The estimand is
$$\theta_q \;=\; \frac{\sum_{i\in S_q} w^q_i a^q_i}{\sum_{i\in S_q} w^q_i}.$$

Examples:
- **E1:** $w\equiv 1$, $a_i = \text{MASE}_i(\text{Naive}) - \text{MASE}_i(\text{HA})$.
- **E2 (pooled within-entity AR(1) slope):** $\hat\rho = \sum_i S^{xy}_i / \sum_i S^{xx}_i$, i.e. $w_i = S^{xx}_i$ and $a_i = S^{xy}_i/S^{xx}_i$.
- **E3:** each of the 15 pairwise MASE gaps is an E1-type estimand.

**Kept set and reference-valued quantities.** Let $K = S_p \cap S_r$. All quantities below use **reference** values $(a^r, w^r)$. Write $w(A) = \sum_{i\in A} w^r_i$, $U_c = c^{-1}(c)$, and:

| Symbol | Definition |
|---|---|
| $W_{rc}$ | $w(S_r\cap U_c)/w(S_r)$ |
| $\mu_{rc}$ | weighted mean of $a^r$ over $S_r\cap U_c$ |
| $W_{kc}$ | $w(K\cap U_c)/w(K)$ |
| $\mu_{kc}$ | weighted mean of $a^r$ over $K\cap U_c$ (set to $\mu_{rc}$ when $W_{kc}=0$) |
| $\theta_{\text{sel}}$ | weighted mean of $a^r$ over $K$ |

**Decomposition terms.**
$$T_1 = \sum_c (W_{kc}-W_{rc})\,\mu_{rc},\qquad T_2=\sum_c W_{kc}(\mu_{kc}-\mu_{rc}),\qquad T_3=\theta_p-\theta_{\text{sel}}.$$

---

## Proposition 1 (exact identity; no-refit forecast)

**(a)** $\theta_p-\theta_r = T_1+T_2+T_3$ for every pair of pipelines.

**(b)** Suppose the candidate is **selection-only on evaluated entities**: $S_p\subseteq S_r$, and for every $i\in S_p$, $a^p_i=a^r_i$ and $w^p_i=w^r_i$. Then $T_3=0$, so
$$\theta_p-\theta_r = T_1+T_2 = \theta_{\text{sel}}-\theta_r.$$
The right-hand side depends only on the reference per-entity results $(a^r,w^r)$, the cells $c$, and the set $S_p$. No model is refitted on the candidate panel.

**Proof.**
- (a) Weighted means aggregate over a partition. Hence $\theta_r=\sum_c W_{rc}\mu_{rc}$ and $\theta_{\text{sel}}=\sum_c W_{kc}\mu_{kc}$.
- Add and subtract $\sum_c W_{kc}\mu_{rc}$:
$$\theta_{\text{sel}}-\theta_r=\sum_c (W_{kc}-W_{rc})\mu_{rc}+\sum_c W_{kc}(\mu_{kc}-\mu_{rc}) = T_1+T_2.$$
  A cell with $W_{kc}=0$ contributes $0$ to $T_2$ by the convention above.
- Adding $T_3=\theta_p-\theta_{\text{sel}}$ to both sides gives (a).
- (b) Under the hypothesis $K=S_p$, and the candidate's values on $K$ coincide with the reference's. So $\theta_p=\theta_{\text{sel}}$ and $T_3=0$. $\square$

**When does (b) hold in practice?** It needs two things:
1. *The pipeline only removes entity-periods or whole entities and leaves the retained series unchanged.*
2. *Each entity's evaluation depends only on its own series.*

Condition 2 holds for per-entity MASE and for the within-entity $S^{xy},S^{xx}$ of E2. Then $S_p$ is computable from the processed panel alone: it is the set of entities whose series pass the evaluator's per-entity filter (here length $\ge 16$).

The pre-registered checks (D2, P1, N3, Q7) confirm $T_3=0$ to machine precision for every selection bug, on every panel and every estimand.

**Corollary 1 (rankings).** A ranking of $M$ models is a function of the $\binom M2$ pairwise gaps, each of which is an E1-type estimand. Under the hypothesis of (b), every gap is forecast exactly, so the candidate's ranking is forecast exactly without refitting. (Observed: 16/16 cells in P1, and every EIA cell in N3.)

**What Proposition 1 does *not* cover.** An *alteration* (forward-fill, interpolation, winsorizing, target shift, scale leak) changes $a^p_i$ for retained $i$, so $T_3\neq 0$. No function of $(a^r,w^r,c,S_p)$ alone can recover it. This is the formal reason the ledger and decomposition are blind to alteration errors (B3, B5, x5, x7) and to interpolated rows. The paper should state this scope limit next to the proposition.

---

## Proposition 2 (composition term: when a reweighting can matter)

Let $\mathrm{TV}=\tfrac12\sum_c|W_{kc}-W_{rc}|$, $\bar\mu_r=\theta_r=\sum_c W_{rc}\mu_{rc}$, and $\operatorname{range}(\mu_r)=\max_c\mu_{rc}-\min_c\mu_{rc}$, both maxima over cells with $W_{rc}>0$. Assume $W_{kc}>0\Rightarrow W_{rc}>0$; this always holds when $K\subseteq S_r$.

**(a) Total-variation bound (sharp).**
$$|T_1|\;\le\;\mathrm{TV}\cdot\operatorname{range}(\mu_r)\;\le\;2\,\mathrm{TV}\cdot\max_c|\mu_{rc}-\bar\mu_r|.$$
The first inequality is attained when all reweighted mass moves from an arg-min cell to an arg-max cell.

**(b) χ² bound (composition × heterogeneity).** With $\chi^2=\sum_c W_{rc}\,(W_{kc}/W_{rc}-1)^2$ and the between-cell standard deviation $\sigma_B=\big(\sum_c W_{rc}(\mu_{rc}-\bar\mu_r)^2\big)^{1/2}$,
$$|T_1|\;\le\;\sqrt{\chi^2}\;\sigma_B.$$

**(c) Coverage-only condition.** If $\mu_{rc}$ is constant on the set of cells whose weight changes ($W_{kc}\ne W_{rc}$), then $T_1=0$, however large TV is.

**Proof.**
- (a) Since $\sum_c W_{kc}=\sum_c W_{rc}=1$, for any constant $m$ we have $T_1=\sum_c (W_{kc}-W_{rc})(\mu_{rc}-m)$.
- Split the sum into cells with $W_{kc}>W_{rc}$ (total excess $=\mathrm{TV}$) and cells with $W_{kc}<W_{rc}$ (total deficit $=\mathrm{TV}$). Take $m=0$. Then
$$T_1\le \mathrm{TV}\max_c\mu_{rc}-\mathrm{TV}\min_c\mu_{rc}=\mathrm{TV}\operatorname{range}(\mu_r),$$
  and symmetrically $T_1\ge-\mathrm{TV}\operatorname{range}(\mu_r)$.
- Equality: put all excess on one arg-max cell and all deficit on one arg-min cell.
- The second inequality uses $\operatorname{range}\le 2\max_c|\mu_{rc}-\bar\mu_r|$, because $\bar\mu_r$ lies in $[\min,\max]$.
- (b) Write $T_1=\sum_c W_{rc}\,(W_{kc}/W_{rc}-1)(\mu_{rc}-\bar\mu_r)$, using $\sum_c W_{rc}(W_{kc}/W_{rc}-1)=0$. Then apply Cauchy–Schwarz in $L^2(W_r)$.
- (c) Use $m$ equal to the common value of $\mu_{rc}$ on the changed cells; every remaining term has $W_{kc}=W_{rc}$. $\square$

**Interpretation.** A large composition shift (TV, χ²) is **necessary but not sufficient** for a large between-cell effect. The other factor is outcome heterogeneity *along the coordinate the bug selects on*.
- The P12 grid bug selects on entry calendar quarter.
- In the bank panels, entry-quarter cohorts have nearly equal $\mu$. So the bug is a coverage error despite TV up to 0.41.
- In the EIA power-plant panel, the cohorts differ: $\mu$ ranges from $-0.57$ (incumbents) to $+0.02$ (Q3–Q4 entrants). The same code is a bias error (17%).

Proposition 2 is the formal version of "harmlessness is a property of the panel, not of the code." It also tells the auditor *what to measure*: $\operatorname{range}(\mu_r)$ or $\sigma_B$ along the selected coordinate, from the reference run alone.

---

## Proposition 3 (within-cell selection term)

For cell $c$ with $W_{rc}>0$, let:
- $\pi_c=w(K\cap U_c)/w(S_r\cap U_c)$, the weighted retention rate;
- $k_i=\mathbf 1\{i\in K\}$;
- $\sigma_{rc}$, the weighted standard deviation of $a^r$ within $S_r\cap U_c$.

**(a)** $\displaystyle \mu_{kc}-\mu_{rc}=\frac{\operatorname{Cov}_c(a^r,k)}{\pi_c}$, where $\operatorname{Cov}_c$ is the $w^r$-weighted covariance over $S_r\cap U_c$. Hence
$$|T_2|\;\le\;\sum_c W_{kc}\,\sigma_{rc}\sqrt{\frac{1-\pi_c}{\pi_c}}.$$

**(b) Cell-measurable selection.** If retention is constant within each cell ($k$ is a function of $c$), then $T_2=0$ and the whole selection effect is $T_1$.

**Proof.**
- (a) $\mu_{kc}=\mathbb E_c[a\,k]/\mathbb E_c[k]$ with $\mathbb E_c[k]=\pi_c$. So $\mu_{kc}-\mu_{rc}=(\mathbb E_c[ak]-\mathbb E_c[a]\mathbb E_c[k])/\pi_c$.
- $|\operatorname{Cov}_c(a,k)|\le\sigma_{rc}\sqrt{\pi_c(1-\pi_c)}$ by Cauchy–Schwarz, because $k$ is Bernoulli with mean $\pi_c$.
- (b) $\operatorname{Cov}_c(a,k)=0$ when $k$ is constant on the cell. $\square$

**Consequence for audit design.** If the cells are aligned with the coordinate the bug selects on, $T_2$ vanishes and $T_1$ carries everything. Observed examples:
- B1 (entry cohort) and B2 (exit status) give $T_2=0$ exactly.
- x4 (a history-length filter is not cell-measurable under entry×exit cells) gives a large $T_2$.

When an auditor does not know the selection coordinate, a large $T_2$ relative to $T_1$ signals that the cells miss it. Re-cutting the cells (by age or history length, or by technology for EIA, where technology cells carry 64% of B1's selection effect) is then the right move.

---

## Proposition 4 (minimum-length filters: shift formula and dose-response)

Let $n_i$ be entity $i$'s evaluable series length, let the evaluator require $n_i\ge L_0$ (here $L_0=16$), and let the candidate additionally require $n_i\ge L$. Write:
- $\pi_L=w(\{n\ge L\})/w(S_r)$;
- $\mu_{\ge L}$ and $\mu_{<L}$, the reference weighted means of $a$ over $\{n\ge L\}$ and $\{L_0\le n<L\}$;
- $\theta(L)=\mu_{\ge L}$.

Assume the filter does not alter retained series, as in Proposition 1(b).

**(a)** For $L\le L_0$, $\theta(L)=\theta_r$, i.e. no effect. For $L>L_0$,
$$\theta(L)-\theta_r=(1-\pi_L)\,(\mu_{\ge L}-\mu_{<L}).$$

**(b) Dose-response.** If $m(n)=\mathbb E_w[a\mid n]$ is monotone in $n$, then $\theta(L)$ is monotone in $L$, with the same direction. In particular the sign of $\theta(L)-\theta_r$ is constant in $L$.

**Proof.**
- (a) For $L\le L_0$ the two filters are nested and the candidate keeps exactly $S_r$.
- For $L>L_0$, $\theta_r=\pi_L\mu_{\ge L}+(1-\pi_L)\mu_{<L}$. So $\theta(L)-\theta_r=(1-\pi_L)(\mu_{\ge L}-\mu_{<L})$.
- (b) Take $m$ nondecreasing. Then $\theta(L)=\mathbb E[a\mid n\ge L]$ is a weighted average of $m(n)$ over $n\ge L$. Raising $L$ removes the smallest values of $m$ from this average, so it cannot decrease. Formally, $\theta(L+1)-\theta(L)=\frac{P(n=L)}{P(n\ge L+1)}\big(\theta(L)-m(L)\big)\ge 0$, since $\theta(L)\ge m(L)$ when $m$ is nondecreasing. $\square$

**Interpretation.** The ablation's finding has a closed form. Every material LLM divergence is a minimum-length choice $L\in\{20,24\}$ above the evaluator's $L_0=16$. Its effect is
$$(\text{share of evaluated weight with } L_0\le n<L)\times(\text{outcome gap between long- and short-history entities}).$$
- The shift is exactly 0 at $L=L_0$.
- Its monotone dose-response in $L$ (banks: up to 35%; plants: up to 82% at $L=48$) follows whenever the forecast advantage of HA over Naive varies monotonically with history length.
- Both inputs to the formula come from the reference run. So the harm of *any* proposed minimum length can be tabulated before the filter is applied.

---

## Status and checks

- Propositions 1, 3(b) and 4(a) are identities; the pre-registered results already confirm them numerically (T3 = 0, T2 = 0 for cell-measurable bugs, L16 = 0).
**Numerical checks run on hanqing-win** (outputs in `results_win/proofs/`):

| Check | Scope | Result |
|---|---|---|
| Identity of Prop 1, and $\theta_{\text{sel}}-\theta_r=T_1+T_2$ | 4 selection bugs × 5 panels (4 bank + EIA) | holds in all 20 |
| Bounds 2(a), 2(b), 3(a) | same 20 cases | hold in all 20 |
| Formula 4(a) vs measured minimum-length shift | 5 panels × 10 values of $L$ | max error **8e-16** |

Scripts: `scripts/verify_props.py` (identity and bounds) and `scripts/verify_prop4.py` (formula 4(a)).

On the bounds:
- **They are informative, not tight.** For example, for B1 on EIA, $|T_1|=0.043$ against 0.189 (TV × range) and 0.149 (χ²·σ_B).
- **The χ² form is uniformly tighter than TV × range** on these panels.
- **They separate the cases correctly.** B1's bound is small on every bank panel, because $\sigma_B$ along entry quarter is small. On EIA the bound reaches 0.149, consistent with B1 being material there.

On formula 4(a): the reference is the task-1 spec reference used by the ablation. An earlier check against the unfiltered reference gave errors up to 0.034; that was a population mismatch, not a failure of the formula.

**Monotonicity premise of 4(b)** (`prop4_m_of_n.csv`): $m(n)$ falls with history length on both cash panels and both BHC panels, with small local reversals. It is flat on bank_leverage until 63 quarters, and it changes sign on EIA (+0.05 at 16–19 quarters, −0.58 at 64–95). The premise therefore holds approximately, not exactly.
- The measured dose-response is still monotone on every panel, because the formula averages $m(n)$ over the whole tail $n \ge L$.
- The paper should report $m(n)$ by length bin and state 4(b) as a sufficient condition, not as a fact about the data.
