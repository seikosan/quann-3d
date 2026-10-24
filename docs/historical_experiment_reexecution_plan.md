# Historical Experiment Re-execution Plan

**Status:** Proposed; run only after the fixes are verified in the exact code revision used for reruns.  
**Purpose:** Replace or qualify historical results affected by resume-state, data-split, EOS, point-head, pooling, and QUANN search/checkpoint defects. Keep old logs and checkpoints immutable; write reruns to fresh directories and new append-only logs.

## Executive summary

Historical PPL values are not uniformly comparable: several evaluations sampled training data or overlapped it, BPE targets omitted EOS, and CloudLM's point head lacked a learning signal. QUANN results also require an exposure audit for candidate-search direction/sigma and checkpoint resume. Re-run claim-bearing matched comparisons from scratch with fixed corpus/BPE/splits/seeds, then refresh the research and paper/patent evidence from the new results. Do not resume old checkpoints for these re-evaluations.

The first priorities are (1) R3 200K max-vs-attention, (2) the D32/D64 capacity comparison, (3) the 20K D32/D64/D128 plus baseline comparison, and (4) all QUANN runs cited by `REVISION.md` or formal claims. Historical runs can remain in the record but must be labelled legacy when superseded.

## Preconditions and common protocol

1. Confirm the seven fixes are present in one recorded revision; the current worktree has uncommitted edits, so capture the exact tested commit/hash before launching. Run CloudLM/tokenizer tests and `python3 -m pytest quann_cloud/tests/ -v` before experiment runs.
2. For every run record corpus file/hash, exact train/holdout row ranges, tokenizer/BPE hash, EOS behavior, seed, model/config, code revision, checkpoint path and evaluation command. Make train and evaluation sets disjoint; for the 200K wiki protocol reserve `[200000, 202000)` as the shared 2K holdout if corpus availability confirms those rows are valid.
3. Use fresh checkpoints for every re-execution. Never continue a historical checkpoint across the EOS, point-head, pooling, or QUANN search changes. For any future intentional resume, test equality against uninterrupted training and restore model, optimizer, scheduler, RNG (CPU/device), and sampler state.
4. Use a common tokenizer for each matched comparison; include EOS targets consistently in training and evaluation. Report held-out PPL as primary, train PPL as diagnostic, and disclose any legacy in-sample metric separately. Keep all comparison arms on the same data split, seed, and evaluation code.
5. QUANN: audit each script/config/checkpoint lineage for candidate direction/sigma usage, resume, and eval overlap before assigning final risk. Re-run affected QUANN training from initialization; evaluation-only jobs may be repeated against corrected checkpoints without retraining when their training lineage is unaffected.
6. Preserve existing logs/checkpoints. Use fresh run names/directories and append-only logs. Update `RESEARCH.md`, `quann_cloud/REVISION.md`, and paper/patent claims only after the matched results and audit are complete.

## Risk matrix

Risk is the chance that a reported result or conclusion changes materially, not the probability that the code fails. QUANN ratings are provisional until the exposure audit is complete.

| Experiment / historical result | Risk | Main exposure | Re-execution disposition |
|---|---|---|---|
| R3 200K max vs attention (`run_r3_200k`) | HIGH | Max-pool padding; point-head signal; EOS; holdout validity | Full matched duel, both arms from scratch |
| R1 D32 vs D64 200K, 50 ep (`run_d32_200k`, `run_d64_200k`) | HIGH | Point-head/EOS; prior PPL protocol; capacity conclusion depends on matched eval | Both models from scratch, common BPE and holdout |
| R3 max vs attention 20K (`run_r3_20k`) | HIGH | Pooling defect and point-head/EOS; existing disjoint eval still uses old training targets | Re-run both arms; retain old duel as legacy |
| R1 20K D32/D64/D128 + causal baseline (`run_rev1_20k`) | HIGH | Baseline resume path, shared EOS/eval protocol, CloudLM head | Re-run all four training arms from scratch and matched holdout |
| Full pipeline (`run_final_pipeline`) | HIGH | Explicit baseline resume; 1.5M CloudLM; old comparison/eval protocol | Rebuild only claim-bearing stages, not blindly execute old pipeline |
| QUANN training / language-model and PPL comparisons | HIGH | Candidate direction/sigma, resume, eval overlap; EOS if CloudLM tokenizer/path is used | Full lineage audit; re-run affected training and all PPL evaluation |
| QUANN real-data/non-PPL results | MEDIUM | Candidate-search/resume exposure; PPL holdout defect may not apply | Audit train lineage; reproduce affected runs; repeat metrics from held-out data if applicable |
| QUANN inference/package/edge benchmarks | LOW–MEDIUM | Usually not affected by training-time search/eval fixes; inherited checkpoint may be | Re-run only if source checkpoint is superseded or benchmark claim is cited |
| Any historical baseline comparison | HIGH | Baseline resume state and overlapping/in-sample evaluation | Retrain baseline, then compare on the shared disjoint holdout |

## Re-execution priority

1. **P0 — claim blockers:** R3 200K matched duel; D32/D64 200K matched duel; baseline comparison(s) supporting current CloudLM-vs-transformer claims; QUANN results explicitly used in `REVISION.md` hypotheses or paper/patent claims.
2. **P1 — interpretive evidence:** R3 20K; `run_rev1_20k` full capacity/baseline comparison; QUANN sensitivity/revision variants and other held-out PPL comparisons.
3. **P2 — scale and pipeline evidence:** claim-bearing 1.5M/full-pipeline stage; remaining QUANN corpus-scale, real-data, and hybrid runs whose training lineage is affected.
4. **P3 — conditional refresh:** inference, packaging, and edge benchmarks. Repeat only when their checkpoint changes or their reported claim needs a corrected source model.

Within each priority, execute one MPS training job at a time. Do the QUANN exposure audit before scheduling P1/P2; do not spend compute on unaffected inference-only measurements.

## Detailed plan

### CloudLM runs

| Run | Risk | Plan and expected impact |
|---|---|---|
| `run_r3_200k` max vs attention, D32/200K/10 ep | HIGH | Re-run max and attention sequentially with identical seed, BPE, split and fresh checkpoints. Point-head/EOS changes may shift both PPLs; corrected max masking may especially improve max pooling when valid activations are negative. The prior 20K max advantage may persist, narrow, or reverse at scale; do not assume direction. |
| `run_d32_200k` and `run_d64_200k`, 50 ep | HIGH | Re-run both with the same 200K wiki rows, common BPE/EOS, matched initialization policy and `[200000,202000)` holdout. Report paired PPL and uncertainty across at least 3 seeds if the capacity difference is used as a strong claim; otherwise mark single-seed evidence exploratory. Revised objective and true holdout can materially change absolute PPL and the small D64-vs-D32 gap. |
| `run_r3_20k` max vs attention, D32/20K/10 ep | HIGH | Re-run as a small-data matched control; the old disjoint holdout corrects overlap only, not old model training defects. Keep its result as secondary to the 200K duel. Attention may continue to overfit or the repaired head/EOS may alter the gap. |
| `run_rev1_20k` D32/D64/D128 + causal baseline | HIGH | Train baseline and all CloudLM sizes from scratch on one fixed 20K split; compare only on a held-out subset excluded from tokenizer training where feasible. Replace old baseline resume and in-sample comparison. Expect all absolute PPLs to move; relative capacity ranking and baseline gap may change. |
| `run_final_pipeline` | HIGH | Decompose into (a) clean 200K causal baseline, (b) a newly matched CloudLM/baseline comparison, and (c) a separate 1.5M D32 training rerun only if the scale result is claimed. Do not replay its `45→50` baseline resume. Revalidate 1.5M PPL on disjoint data and EOS-compatible BPE. |
| `run_d32_200k` / `run_d64_200k` contributions to `RESEARCH.md` | HIGH | Supersede legacy 50-epoch values and R1 conclusion only after the matched paired rerun finishes. Existing legacy PPLs (including 6.45 and 19.08) are not strict held-out comparisons. |

### QUANN-Cloud experiment inventory

Apply the common audit to every listed experiment. **HIGH** means rerun training if it uses QUANN candidate search, resume, or an affected CloudLM/PPL path. **MEDIUM** means verify lineage and repeat claim-bearing metrics; downgrade to LOW only with evidence that the result is independent of all seven fixes. This inventory is based on experiment names and must be finalized by inspecting each runner and its output lineage before execution.

| Experiment | Provisional risk | Required action |
|---|---|---|
| `experiments/quann_training.py` | HIGH | Audit direction/sigma and resume; rerun QUANN training from initialization if either is active; validate its evaluation split. |
| `experiments/real_training.py` | HIGH | Audit search and checkpoint lineage; repeat training if affected and recompute claim-bearing metrics on held-out data. |
| `experiments/full_corpus.py` | HIGH | Audit candidate search, resume, and any PPL/eval partition; rerun affected training and held-out evaluation. |
| `experiments/tiny_corpus.py` | HIGH | Same checks; if it supports a hypothesis, rerun with fixed seed/split and report variance. |
| `experiments/control_20k_single.py`, `compare_20k.py` | HIGH | Recreate controlled arms and comparison from fresh checkpoints; ensure evaluation is disjoint and not corpus tail sampled from training. |
| `experiments/revision_variants.py` | HIGH | Re-run every variant cited by `REVISION.md` where search direction/sigma or resume is used; preserve matched protocol across variants. |
| `experiments/sensitivity_analysis.py` | HIGH | Reproduce any sensitivity conclusion whose candidate search used direction/sigma; distinguish geometry-only analyses of fixed outputs from training-dependent results. |
| `experiments/transformer_vs_qcloud.py`, `compare_final.py` | HIGH | Rebuild both sides if either resumes or uses an affected tokenizer/eval; compare only on the same disjoint holdout. |
| `experiments/small_model_quann.py` | HIGH | Audit training and checkpoint lineage; rerun cited QUANN/PPL findings from scratch if affected. |
| `experiments/hybrid_training.py`, `full_hybrid.py` | HIGH | Audit the QUANN-trained component, resume, and evaluation partition; retrain the affected component and recompute final metrics. |
| `experiments/real_data_test.py` | MEDIUM | Candidate search/resume audit; PPL-tail issue applies only if it reports language-model PPL. Repeat any claimed score whose model lineage is affected. |
| `experiments/edge_benchmark.py` | LOW–MEDIUM | No retraining if inference-only and checkpoint lineage is unaffected; benchmark again against a replacement checkpoint when claim-bearing. |
| `experiments/package_cloudlm.py`, `train_cloudlm.py` | MEDIUM | Audit whether the run trains or packages a legacy CloudLM checkpoint; re-run training if affected, otherwise refresh package/evaluation after source checkpoint replacement. |
| `experiments/xor_gate/xor.py` | LOW–MEDIUM | Confirm it is independent of qcloud trainer/search/resume; keep if isolated, otherwise rerun the affected training. |

For each QUANN script record one outcome: `rerun training`, `reevaluate only`, or `unaffected (evidence)`. Add the script, source checkpoint, metric, split, and decision to the experiment audit before changing `REVISION.md` hypotheses.

### Baseline experiments

Inventory every `train_baseline.py` run that is a comparator or contributes to a cited result, including `run_rev1_20k` and the baseline stage in `run_final_pipeline`. Re-execute from initialization with corrected optimizer/scheduler/RNG handling, shared EOS/tokenizer and a disjoint held-out set. Historical resumed baselines are not valid substitutes. Any evaluation-only comparison against an old baseline checkpoint is invalidated if that checkpoint's training used resume or incompatible targets.

## Resource budget and timeline

Estimates below are **provisional single-MPS accelerator-hours**, not measured runtimes. MPS throughput, preprocessing, eval, and the QUANN algorithm cost are unverified. Measure the first completed run's elapsed time and update remaining rows before launching a large queue. No two heavy MPS trainings should overlap.

| Work package | Runs | Estimated MPS hours |
|---|---:|---:|
| CloudLM 20K: R3 pair + Rev1 4 arms | 6 trainings | 3–12 h |
| CloudLM 200K: R3 pair, 10 ep | 2 trainings | 8–24 h |
| CloudLM 200K: D32/D64, 50 ep | 2 trainings | 30–100 h |
| Baseline clean reruns and comparisons (20K + 200K) | 2 trainings + evals | 4–16 h |
| 1.5M D32/20 ep scale stage, conditional | 1 training | 25–100 h |
| QUANN audit + claim-bearing retraining/re-evaluation | up to 17 scripts; audit will reduce run count | 20–120 h |
| **Total, including conditional stages** | | **90–372 h** |

The QUANN count is an upper-bound inventory, not a commitment to run all 17 scripts. Audit first; unaffected packaging and inference jobs should cost only their short evaluation/benchmark time, excluded from the broad training range. If only P0 is funded, reserve approximately **45–150 MPS hours**. These wide ranges should be replaced with measured per-run estimates after the first 20K and 200K pilots.

| Phase | Calendar estimate | Exit gate |
|---|---|---|
| Fix verification, lineage audit, split/BPE lock | 2–4 working days | Tests pass; scripts/checkpoints mapped; protocol frozen |
| P0 matched reruns | 1–2 weeks, serial MPS | R3 200K, D32/D64, baseline and cited QUANN results evaluated |
| P1 sensitivity and 20K comparisons | 3–7 days | 20K controls and cited variants reconciled |
| P2 scale reruns and evidence refresh | 1–2 weeks | 1.5M/remaining QUANN claims verified or explicitly retired |
| Reporting and claim updates | 1–3 working days | `RESEARCH.md`, `REVISION.md`, and paper/patent references agree |

## Result acceptance and claim updates

- Publish no single legacy PPL as held-out. Include exact split boundaries, sample count, BPE/EOS version, and seed in every result row.
- A claimed comparison must use matched conditions and valid holdout metrics on both arms. If a result reverses or the rerun is unaffordable, mark the old claim `legacy/unverified` rather than implying confirmation.
- Update `RESEARCH.md`'s “Son Deneyler” only after a run completes; keep the live-run table accurate. Update `quann_cloud/REVISION.md` hypotheses from the audit plus rerun evidence, not from the expected-impact predictions above.
- Paper/patent claims must cite the corrected experiment and state the tested scope. Preserve prior logs/checkpoints as historical evidence; never overwrite them.
