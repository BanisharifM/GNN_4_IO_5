# GraphXIO (IEEE Cluster 2026) — Camera-Ready Action Items

Created 2026-06-22. Updated 2026-06-23 to map the FINAL submitted rebuttal. Internal working
doc (LOCAL ONLY, not paper-tied, not for Overleaf).
Notification: July 5. If accepted, every promise in the rebuttal becomes a task below.
Rebuttal SUBMITTED and received (PC-chair confirmation email, A. Butt) at commit `f995b5e`
(GraphXIO_Papers), 792 words. The final references line is "After correction, every flagged
entry cites a real, published paper"; the speedup provenance is "Darshan logs and benchmark
and application run records" (NOT "SLURM records" — SLURM appears nowhere in the paper).

The single organizing principle: **unify the entire paper onto ONE evaluation run
(the production-arch run), end to end.** Every number issue below traces back to the
paper mixing two different runs.

---

## 0. CRITICAL — the run-consistency problem (this is R3's catch, confirmed in our data)

The reported numbers come from two different evaluation runs that were never unified.

| Source | GAT RMSE | LightGBM RMSE | reported delta |
|---|---|---|---|
| Table III / abstract (paper) | **0.2370** | **0.2892** | 18.0% |
| Saved bootstrap artifact `bootstrap_ci_prod_k30.json` | **0.2568** | **0.2857** | 10.1% |

- `0.2370` is a REAL run: `RESULTS_TABLE.md` line 62, production-arch k=30, ep219,
  R2 0.9437, MAE 0.1621, "DONE 2026-05-07". It beats LightGBM. NOT fabricated.
- But the paired bootstrap (the source of the 10.1% claim) was computed on a
  DIFFERENT GAT checkpoint (0.2568) and a DIFFERENT LightGBM (0.2857).
- The paper's LightGBM `0.2892` also does not match the saved LightGBM `0.2857`
  (`lgb_currentCSV_metrics.json`, `RESULTS_TABLE.md` line 17). Two different retrainings.
- Result: no saved artifact ties 0.2370/0.2892 to a bootstrap CI. The table and the
  CI are from different runs. That is precisely what R3 flagged.

### Action
1. Pick ONE run for the whole paper. Recommend the production-arch k=30 run (0.2370).
2. Recompute the paired bootstrap on THAT run's predictions, on the reported test split.
   Expect the headline percentage to land near ~17% (0.2370 vs 0.2857), NOT 10.1%.
   Decide: report ~17% point delta + matching bootstrap, OR keep the 0.2568 run as
   headline at ~10%. Either is fine. Pick one and use it everywhere (abstract, Table III,
   text on eval.tex lines 65, 87, 97, 112).
3. Fix LightGBM to the single retrained value (0.2857 if you use the saved current-CSV
   model). Update eval.tex line 65 and the Table III LightGBM* row (currently 0.2892).
4. After unifying: RMSE, R2, MAE, CI, and every percentage must trace to the same run.

The rebuttal already commits to this ("present one consistent set"). No rebuttal edit
needed. This section is the camera-ready execution of that promise.

---

## 1. Artifact-release hygiene (before releasing anything)

The rebuttal's exact promise (`f995b5e`): release **code, data, trained models, training
logs, result files, and closed-loop measurement records** for independent reproduction.
Deliver all six. The closed-loop measurement records must include the IOR and E2E
(benchmark and application) run records that back the 11.58x / 4.24x speedups, since the
rebuttal names "Darshan logs and benchmark and application run records" as the provenance.

There are TWO architecture families in `results/`:

- **production-arch** (keep): `bootstrap_ci_prod_*.json`, GAT ~0.2555-0.2596, beats
  LightGBM (+9% to +10.6%). `predictions_gat_prod_*.npz`. Production no-aug k=50 = 0.2410.
- **old-arch / earlier pipeline** (QUARANTINE): `bootstrap_ci_k20/k30/k50/k100.json`,
  `bootstrap_ci_k50_noaug.json`, GAT ~0.305-0.314, **LOSES** to LightGBM (-7% to -9.8%).
  `bootstrap_ci_existing.json` is broken (LightGBM = 1.357). Do NOT ship these.

### Action
- Release only the production-arch pipeline and its predictions.
- A reviewer or AD checker who runs the old-arch files sees GAT losing. Remove them
  from any released artifact tree.

---

## 2. Table III baseline rows (rebuttal conceded these are wrong)

Per `INTEGRITY_SWEEP.md`, the GNN-baseline rows are unreliable:
- GCN row 0.3085: no completed run (training stopped at Val RMSE 0.2523, still falling).
- GraphSAGE row 0.2886: real completed run = 0.2579.

### Action
- Recompute GCN and GraphSAGE from a completed run, or mark them honestly. Do not keep
  0.3085 / 0.2886. The GAT-vs-LightGBM headline does not depend on these rows.
- Re-check the per-row R2-vs-RMSE consistency R3 flagged once the rows are recomputed.

---

## 3. Full camera-ready checklist (rebuttal promise -> task)

Priority: P0 = must, load-bearing. P1 = important. P2 = secondary, only if feasible.

| # | Task | Priority | Effort | Status / notes |
|---|---|---|---|---|
| a | Unify on one run; recompute bootstrap on reported split; one consistent RMSE/R2/CI/% set | P0 | low | data exists; see §0 |
| b | Fix Table III GCN + GraphSAGE rows | P0 | low-med | see §2 |
| c | Remove HNSW claim from contributions | P0 | trivial | deployed method is GPU exact k-NN (51s, 92.7% recall). Do NOT write ">99% recall" |
| d | Rescope latency to prediction; report GNNExplainer/IG explanation cost separately | P0 | low | 3.09 ms is prediction-only; measure the explanation step once |
| e | Define consensus precisely: per-method top-K, attention->feature mapping, z-score 2-of-3 | P0 | low (writing) | no experiment, just specify |
| f | Add AI-assistance acknowledgment (tools + role) | P0 | trivial | draft sits commented in main.tex; uncomment + finalize. Must be TRUE about scope |
| g | Ship corrected bibliography | P0 | done-ish | `references_fixed.bib` ready; strip the "was: [wrong]" comments before camera-ready |
| h | Top-1/top-3 diagnosis accuracy table (consensus vs each single method vs SHAP-on-LightGBM vs Drishti) on labeled IOR/IO500 | P1 | med | 3 reviewers asked. Expect MODEST numbers. Report honestly under the "reliability filter / ranked candidates" framing. Do NOT fabricate; real IOR consensus hit@k was 0 |
| i | Sensitivity analysis: 2-of-3 threshold + z-score aggregation | P1 | med | R4 asked |
| j | 45-vs-49 ablation | P2 | low-med | production no-aug exists (k=50, 0.2410, beats LightGBM). Use production runs, not old-arch noaug. For a clean k-matched table, run one matched k. Honest story: augmentation helps a little, both beat baseline |
| k | LightGBM + neighbor-features ablation | P2 | med-high | R3/R4 asked; strongest answer to "does the graph help". Genuine new run, outcome unknown (may narrow the gain). Keep only if deliverable |
| l | Scope the Cori claim (Cray XC40 Lustre corpus; local retraining = deployment path; cross-facility = future work). Drop "leadership-class" wording (Cori retired 2023) | P1 | low (writing) | R1 |
| m | Remove the stray GraphSAGE-proxy / K-ablation sentence | P0 | trivial | R1 quoted it. It also leaks prior-submission history. Delete entirely. K-ablation Table IV is on the production GAT |
| n | Revise Figure 4 for legibility (legend overlaps bars) | P1 | low | R4. "revise", not "regenerate" |
| o | Define notation on first use (a_ij, top-K, counter names, acronyms: E2E/HPC/IOR/GAT/k-NN/HNSW/MAE/RMSE) | P1 | low | R2 |
| p | Restructure methodology: (i) method, (ii) implementation/training, (iii) evaluation; move HPO/optimizer to reproducibility; add a related-work gap paragraph | P1 | med | R2 |
| q | Future-work paragraph: incremental graph maintenance, continual retraining, multi-GPU construction, coupling with rule-based tools | P2 | low | R1/R2 |
| r | State the gain belongs to the **attention-weighted relational model, not the graph alone** (eval + discussion wording) | P0 | trivial (writing) | rebuttal promised this exact framing to R3/R4; GCN underperforms, so never claim "graph structure alone" drives the gain |
| s | Scope the E2E case to **one application**; broader application workloads = future work | P1 | trivial (writing) | rebuttal conceded "covers one application" |
| t | Release the full artifact exactly as promised: code, data, trained models, training logs, result files, closed-loop measurement records (incl. IOR + E2E run records) | P0 | med | see §1; deliver all six, production-arch only |

---

## 4. Integrity guardrails (do NOT re-assert)

From `INTEGRITY_SWEEP.md`. Never put these back in any version:
- consensus accuracy / hit@k numbers (real IOR hit@k = 0)
- GCN 0.3085, GraphSAGE 0.2886
- IO500 consensus sum 4.02 (real top feature = nprocs, 2.24)
- manual E2E 46.21 -> 158.43 / 3.4x (no source file)
- ">99% k-NN recall" (real 92.7%)
- old-arch GAT numbers where GAT loses (~0.30)

Never blame the AI for any error. Never mention the prior submission venue or history.

---

## 5. Evidence appendix (traceability)

Saved bootstrap point estimates (`results/bootstrap_ci_*.json`, n_test=200000, 10k resamples):

| file | GAT RMSE | LGB RMSE | rel % | verdict |
|---|---|---|---|---|
| prod_k20 | 0.2577 | 0.2857 | +9.80 | production, GAT wins |
| prod_k30 | 0.2568 | 0.2857 | +10.13 | production, GAT wins (current bootstrap basis) |
| prod_k50_int | 0.2555 | 0.2857 | +10.60 | production, GAT wins |
| prod_k100_int | 0.2596 | 0.2857 | +9.14 | production, GAT wins |
| k20 | 0.3058 | 0.2857 | -7.03 | OLD-ARCH, GAT loses |
| k30 | 0.3068 | 0.2857 | -7.38 | OLD-ARCH, GAT loses |
| k50 | 0.3084 | 0.2857 | -7.94 | OLD-ARCH, GAT loses |
| k100 | 0.3113 | 0.2857 | -8.95 | OLD-ARCH, GAT loses |
| k50_noaug | 0.3137 | 0.2857 | -9.78 | OLD-ARCH, GAT loses |
| existing | 0.2577 | 1.3572 | +81.0 | BROKEN (LGB value invalid) |

Paper (`05_evaluation.tex`): Table III GAT 0.2370 / R2 0.9437 / MAE 0.1621 / +18.0%;
LightGBM* 0.2892; GraphSAGE 0.2886. Text: 10.1% bootstrap + 18.0% point delta (line 97).
k-ablation Table IV: k=20 0.2384, k=30 0.2370 (lines 129-130).
`RESULTS_TABLE.md`: line 17 LightGBM 0.2857; line 62 GAT k=30 0.2370 (production, DONE);
line 63 / 228 production no-aug k=50 0.2410.
