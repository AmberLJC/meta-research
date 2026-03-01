# Experiment Design Reference: Principles of Rigorous Experimentation

Every hypothesis must be verified by experiment. No claim enters the research tree as
"supported" without empirical evidence obtained through a well-designed test. This
reference collects the foundational principles that separate rigorous experiments from
ad-hoc exploration.

Use this document as a checklist and conceptual guide when designing experiments in the
[Experiment Design phase](experiment-design.md). Each principle includes a definition,
why it matters, how to apply it, and common violations to watch for.

---

## Table of Contents

1. [Falsifiability](#1-falsifiability)
2. [Controls](#2-controls)
3. [Randomization](#3-randomization)
4. [Blinding and Bias Reduction](#4-blinding-and-bias-reduction)
5. [Replication](#5-replication)
6. [Pre-Registration](#6-pre-registration)
7. [Statistical Power and Sample Size](#7-statistical-power-and-sample-size)
8. [Internal Validity](#8-internal-validity)
9. [External Validity](#9-external-validity)
10. [Construct Validity](#10-construct-validity)
11. [Operationalization](#11-operationalization)
12. [Minimal Sufficiency](#12-minimal-sufficiency)
13. [Independence of Observations](#13-independence-of-observations)
14. [Effect Size and Practical Significance](#14-effect-size-and-practical-significance)
15. [Multiple Comparisons](#15-multiple-comparisons)
16. [Reproducibility](#16-reproducibility)
17. [Transparency and Complete Reporting](#17-transparency-and-complete-reporting)
18. [Ethical Design](#18-ethical-design)
19. [Quick-Reference Checklist](#19-quick-reference-checklist)
20. [Recommended Reading](#20-recommended-reading)

---

## 1. Falsifiability

**Principle**: Design experiments to potentially disprove the hypothesis, not to confirm it.

**Why it matters**: A hypothesis that cannot be disproved by any conceivable observation
is not scientific. Experiments that can only produce confirming evidence are uninformative —
they tell you nothing about whether the hypothesis is actually wrong.

**How to apply**:
- Before designing the experiment, explicitly state: "I would abandon this hypothesis if
  I observe [specific outcome]"
- Define the null hypothesis (H₀) — the default state of the world where your claim is false
- Choose metrics and thresholds where failure is a real possibility
- Include negative controls (conditions where the method SHOULD NOT work)

**Common violations**:
- Designing experiments where every possible outcome "supports" the hypothesis
- Using evaluation metrics that only measure one direction (e.g., recall without precision)
- Post-hoc reinterpreting failed results as "partial support"
- Choosing tasks so easy that any method succeeds (ceiling effects)

**Key reference**: Popper, K. (1959). *The Logic of Scientific Discovery*. Routledge.

---

## 2. Controls

**Principle**: Every experiment needs properly designed control conditions that isolate
the effect of the variable being tested.

**Why it matters**: Without controls, you cannot attribute observed effects to your
intervention. The difference between "X works" and "X works *because of the specific
thing we claim*" depends entirely on controls.

**Types of controls**:

| Control Type | Purpose | Example |
|-------------|---------|---------|
| **Baseline control** | Strong existing method under identical conditions | GPT-4 on the same task with the same prompts |
| **Ablation control** | Your method minus one component | Full model minus the attention mechanism |
| **Negative control** | Condition where the effect SHOULD NOT appear | Random labels, shuffled inputs |
| **Positive control** | Condition where the effect SHOULD clearly appear | Known easy cases that validate the pipeline works |
| **Placebo control** | Identical procedure without the active ingredient | Same architecture with random initialization, no training |

**How to apply**:
- List every claim your experiment makes
- For each claim, ask: "What control eliminates the most plausible alternative explanation?"
- Run baselines under identical conditions (same hardware, same data, same hyperparameter
  budget) — not copied from other papers with different setups
- Include at least one negative control as a sanity check

**Common violations**:
- Comparing against weak or outdated baselines
- Running baselines with different hyperparameter budgets or data splits
- No negative controls (so you cannot detect when your pipeline is broken)
- Missing ablations (so you cannot attribute the effect to specific components)

---

## 3. Randomization

**Principle**: Use random assignment to distribute unknown confounders evenly across
experimental conditions.

**Why it matters**: Systematic assignment (e.g., always putting the hardest examples in
the test set) introduces bias that confounds your results. Randomization ensures that any
unknown factors are equally likely to affect all conditions.

**How to apply**:
- Use random seeds for data splits, model initialization, and training order
- Document all seeds and ensure they are different across runs (variance > 0)
- For data splits: use stratified random sampling when classes are imbalanced
- For hyperparameter search: randomized search often outperforms grid search (Bergstra &
  Bengio, 2012)
- For participant assignment (human studies): random allocation to conditions

**In ML specifically**:
- Random seeds for weight initialization
- Random order of training examples (shuffled batches)
- Random train/val/test splits (or use canonical splits when available)
- Multiple random seeds per condition to measure variance

**Common violations**:
- Using a single random seed (no variance estimate)
- Cherry-picking the best seed for reporting
- Using non-random splits (e.g., first 80% for train, last 20% for test with temporal data)
- Not documenting seeds, making reproduction impossible

**Key reference**: Fisher, R.A. (1935). *The Design of Experiments*. Oliver & Boyd.

---

## 4. Blinding and Bias Reduction

**Principle**: Prevent knowledge of experimental conditions from influencing measurements
and analysis.

**Why it matters**: Researcher expectations unconsciously influence how data is collected,
analyzed, and interpreted. Even in computational experiments, knowing which method is
"yours" biases how you tune, debug, and report results.

**How to apply**:

| Level | What is blinded | When to use |
|-------|----------------|-------------|
| **Single-blind** | Evaluator does not know which condition produced the output | Human evaluation studies |
| **Double-blind** | Neither evaluator nor subject knows the condition | Clinical trials, user studies |
| **Analyst-blind** | Analysis code is written before seeing results | All computational experiments |

**In ML specifically**:
- Write the evaluation script before running experiments
- Pre-commit the analysis plan (see [Pre-Registration](#6-pre-registration))
- When doing qualitative analysis, anonymize which method produced each output
- Use automated metrics where possible to reduce subjective judgment
- Have a collaborator independently verify key results

**Common violations**:
- Iteratively tuning your method while leaving baselines at default settings
- Knowing which outputs belong to your method during qualitative evaluation
- Adjusting evaluation criteria after seeing results

---

## 5. Replication

**Principle**: Results must be reproducible across independent runs, and ideally across
independent implementations.

**Why it matters**: A result that appears only once might be an artifact of a specific
random seed, data ordering, or hardware quirk. Replication separates signal from noise
and establishes the robustness of findings.

**Levels of replication**:

| Level | What varies | What it establishes |
|-------|-------------|---------------------|
| **Same-seed reproduction** | Nothing (exact rerun) | Determinism of the pipeline |
| **Cross-seed replication** | Random seed | Robustness to initialization |
| **Cross-data replication** | Dataset | Generalizability across distributions |
| **Cross-implementation** | Code/team | Independence from implementation details |

**How to apply**:
- Run every condition with multiple seeds (minimum 3, ideally 5+)
- Report mean AND variance (not just the best run)
- Test on multiple datasets when possible
- Verify that your pipeline is deterministic (same seed → same result)
- Document everything needed for independent reproduction

**Common violations**:
- Reporting only the best run out of many
- Running N seeds but only reporting the top K
- No variance estimates in reported metrics
- Claiming generalizability from a single dataset

**Key reference**: Open Science Collaboration (2015). "Estimating the reproducibility
of psychological science." *Science*, 349(6251).

---

## 6. Pre-Registration

**Principle**: Commit to the hypothesis, methods, and analysis plan before collecting or
analyzing data.

**Why it matters**: When analysis decisions are made after seeing data, there are too
many "researcher degrees of freedom" — choices about which metric to report, which
subgroups to highlight, which comparisons to make. Pre-registration eliminates this
flexibility and makes results more credible.

**How to apply**:
- Write the full analysis plan before running large-scale experiments
- Lock the protocol (see [experiment-design.md](experiment-design.md), Step 8)
- Commit the locked protocol to version control BEFORE generating results
- Any analysis not in the pre-registered plan is labeled EXPLORATORY
- Use the [experiment-protocol.md](../templates/experiment-protocol.md) template

**What to pre-register**:
- Primary hypothesis and metric
- Statistical test and decision rule
- Dataset and splits
- Baselines and ablations
- Number of runs/seeds
- Stopping criteria

**Common violations**:
- Writing the analysis plan after seeing results
- Changing the primary metric to whichever shows the best results (outcome switching)
- Adding post-hoc comparisons without labeling them as exploratory
- HARKing: Hypothesizing After Results are Known

**Key reference**: Nosek, B.A., et al. (2018). "The preregistration revolution."
*PNAS*, 115(11), 2600-2606.

---

## 7. Statistical Power and Sample Size

**Principle**: Ensure the experiment has sufficient statistical power to detect the
effect size you care about.

**Why it matters**: An underpowered experiment cannot distinguish between "no effect" and
"effect too small to detect." Running underpowered experiments wastes resources and
produces inconclusive results that contribute to publication bias.

**How to apply**:
- Before the experiment, estimate the minimum effect size that would be meaningful
- Compute the required sample size / number of runs for adequate power (typically 80%)
- If the required sample size is infeasible, acknowledge the limitation explicitly
- For ML: the "sample size" includes number of random seeds, dataset sizes, and number of
  test examples

**Power analysis components**:
```
Required inputs:
- Effect size: minimum meaningful difference (e.g., 2% accuracy improvement)
- Significance level (α): typically 0.05
- Desired power (1-β): typically 0.80
- Variance estimate: from pilot studies or prior work

Output:
- Minimum number of observations / runs needed
```

**Common violations**:
- Running experiments with no consideration of whether they can detect the expected effect
- Claiming "no difference" from an underpowered experiment (absence of evidence ≠ evidence of absence)
- Adding more data or runs only when initial results are not significant (optional stopping)

**Key references**:
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. Erlbaum.
- Button, K.S., et al. (2013). "Power failure." *Nature Reviews Neuroscience*, 14, 365-376.

---

## 8. Internal Validity

**Principle**: Ensure that the observed effect is caused by the independent variable,
not by confounders.

**Why it matters**: Internal validity is the most fundamental requirement. If you cannot
establish that X caused Y (rather than some confounder Z), the experiment has not tested
the hypothesis at all.

**Threats to internal validity in ML**:

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Data leakage** | Test information leaks into training | Strict split discipline, temporal splits |
| **Confounded comparisons** | Methods differ on more than the claimed variable | Ablation controls, matched conditions |
| **Hyperparameter bias** | More tuning budget for your method than baselines | Equal tuning budget across conditions |
| **Compute confound** | Better results due to more compute, not better method | Compare at equal compute/FLOPs |
| **Selection bias** | Non-random selection of examples, datasets, or tasks | Pre-registered dataset selection |
| **Maturation** | Performance improves simply from more training | Fixed training budget or convergence criteria |
| **Instrumentation** | Measurement tool changes between conditions | Same evaluation code for all conditions |

**How to apply**:
- For every observed effect, ask: "What alternative explanation could produce this result?"
- Design ablations that isolate each component
- Match all conditions on compute, data, and tuning budget
- Use the confounders section in the experiment protocol

**Common violations**:
- Training on test data (directly or through repeated evaluation-driven tuning)
- Comparing methods trained with different amounts of data or compute
- Not controlling for model size when comparing architectures
- Ignoring label noise or annotation artifacts

---

## 9. External Validity

**Principle**: Results should generalize beyond the specific experimental setting.

**Why it matters**: An experiment that works only on one dataset, one language, or one
hardware configuration has limited scientific value. External validity determines whether
findings transfer to the broader claims being made.

**How to apply**:
- Test on multiple datasets spanning different characteristics
- Evaluate across domains, languages, or task variants when the hypothesis is general
- Report per-dataset results (not just averages) to show where generalization holds and
  where it fails
- Clearly state the scope of the claim — do not overclaim generality

**Dimensions of generalization**:
- Across datasets (same task, different data)
- Across domains (different task domains)
- Across scales (different model sizes, dataset sizes)
- Across languages or populations
- Across time (temporal distribution shift)
- Across hardware (reproducibility on different GPUs)

**Common violations**:
- Claiming a "general" method based on one benchmark
- Reporting only aggregated metrics that hide per-dataset failures
- Not testing robustness to distribution shift
- Overclaiming from narrow experimental conditions

---

## 10. Construct Validity

**Principle**: The metrics and operationalizations actually measure the theoretical
constructs you claim to study.

**Why it matters**: If your metric does not capture what you claim to measure, your
experiment tests something different from your hypothesis. High accuracy on a benchmark
does not necessarily mean the model "understands" language, "reasons" logically, or
"generalizes" well.

**How to apply**:
- For each metric, ask: "Does optimizing this metric actually mean the hypothesis is
  supported?"
- Use multiple metrics that capture different aspects of the construct
- Include behavioral tests that probe the underlying capability
- Acknowledge the gap between the metric and the theoretical claim

**Common violations**:
- Equating benchmark accuracy with the theoretical construct (e.g., "understanding")
- Using a single metric that can be gamed or shortcut
- Not testing for shortcut learning (e.g., models solving tasks through spurious correlations)
- Conflating correlation metrics with causal claims

**Key reference**: Raji, I.D., et al. (2021). "AI and the Everything in the Whole
Wide World Benchmark." *NeurIPS Datasets and Benchmarks Track*.

---

## 11. Operationalization

**Principle**: Every variable, metric, and condition must have a precise, unambiguous
definition before the experiment begins.

**Why it matters**: Vague definitions create room for post-hoc reinterpretation. When
"performance" is not precisely defined, you can always find a way to claim improvement.

**How to apply**:
- Define every metric with an exact formula (not just a name)
- Specify averaging policies: micro vs macro, per-example vs per-class
- Define edge cases: how are ties handled? what about empty outputs?
- Specify evaluation conditions: temperature, decoding strategy, prompt format
- Document the exact evaluation code (ideally version-controlled)

**Operationalization checklist**:
```
For each metric:
[ ] Name and exact mathematical formula
[ ] Averaging policy (micro / macro / weighted)
[ ] Edge case handling (empty predictions, ties, undefined values)
[ ] Evaluation conditions (deterministic decoding? temperature? top-k?)
[ ] Reference implementation (code path or library + version)
```

**Common violations**:
- Using metric names without definitions ("we measure quality")
- Different evaluation conditions for different methods
- Inconsistent edge-case handling between methods
- Using different library versions that compute metrics differently

---

## 12. Minimal Sufficiency

**Principle**: Design the simplest experiment that can answer the question.

**Why it matters**: Complex experimental setups introduce more potential confounders,
cost more resources, and are harder to debug and interpret. A focused experiment with
clear results is worth more than an ambitious one with ambiguous outcomes.

**How to apply**:
- Start with the cheapest falsification test (see [Judgment Gate](judgment.md), Step 4)
- Add complexity only when simpler experiments cannot distinguish hypotheses
- If a controlled comparison on one dataset suffices, do not run a 20-dataset benchmark
- Separate different questions into different experiments rather than confounding them

**The hierarchy of experimental effort**:
1. **Thought experiment**: Can this hypothesis be eliminated by logic alone?
2. **Back-of-envelope calculation**: Can the effect size be estimated analytically?
3. **Pilot study**: Small-scale test (hours, not days) on a subset
4. **Focused experiment**: Full test of one hypothesis under controlled conditions
5. **Large-scale study**: Multi-dataset, multi-seed, comprehensive evaluation

Move to the next level only when the previous level is inconclusive.

**Common violations**:
- Running massive experiments before checking basic feasibility
- Testing multiple hypotheses in a single confounded experiment
- Using unnecessarily complex setups when a simple comparison would suffice

---

## 13. Independence of Observations

**Principle**: Individual observations should be statistically independent, or dependence
must be explicitly modeled.

**Why it matters**: Standard statistical tests assume independence. When observations are
correlated (e.g., examples from the same document, predictions from the same model run),
standard error estimates are too small and significance tests are unreliable.

**How to apply**:
- Identify the unit of analysis (is it per-example, per-document, per-run?)
- Check for clustering: do examples share sources, authors, time periods?
- Use appropriate tests for non-independent data (paired tests, mixed-effects models,
  clustered standard errors)
- When aggregating, aggregate at the right level (per-document average, not per-token)

**Common violations in ML**:
- Treating per-example metrics as independent when examples come from the same document
- Ignoring temporal correlation in time-series data
- Treating different layers/heads of the same model as independent observations
- Computing confidence intervals from correlated predictions

---

## 14. Effect Size and Practical Significance

**Principle**: Report the magnitude of the effect, not just whether it is statistically
significant.

**Why it matters**: With enough data, any tiny difference becomes "statistically
significant." A 0.1% accuracy improvement with p < 0.001 may be real but meaningless in
practice. Effect size tells you whether the result matters.

**How to apply**:
- Always report the actual difference (e.g., "3.2 accuracy points") alongside any
  statistical test
- Contextualize the effect size: how does it compare to human performance? to the gap
  between existing methods? to the variance across runs?
- Define the minimum meaningful effect size before the experiment
- Use confidence intervals rather than just p-values

**Effect size reporting**:
```
BAD:  "Our method significantly outperforms the baseline (p < 0.05)"
GOOD: "Our method improves accuracy by 3.2 points (82.3 vs 79.1, 95% CI [1.8, 4.6],
       paired bootstrap p < 0.01, N=5 seeds). This closes ~40% of the gap to human
       performance (87.0)."
```

**Common violations**:
- Reporting p-values without effect sizes
- Claiming "significant improvement" for trivially small differences
- Not contextualizing the magnitude of improvement
- Confusing statistical significance with practical importance

**Key reference**: Sullivan, G.M. & Feinn, R. (2012). "Using Effect Size — or Why the
P Value Is Not Enough." *Journal of Graduate Medical Education*, 4(3), 279-282.

---

## 15. Multiple Comparisons

**Principle**: When testing multiple hypotheses simultaneously, account for the increased
probability of false positives.

**Why it matters**: If you test 20 independent hypotheses at α = 0.05, you expect 1
false positive by chance alone. Without correction, running more experiments guarantees
finding "significant" results even when none exist.

**How to apply**:
- Count the total number of comparisons: N methods × M datasets × K metrics
- Apply a correction method appropriate to the setting:

| Method | When to use | How it works |
|--------|------------|--------------|
| **Bonferroni** | Conservative, few comparisons | Divide α by number of comparisons |
| **Holm-Bonferroni** | Less conservative, sequential | Step-down procedure on sorted p-values |
| **Benjamini-Hochberg** | Many comparisons, FDR control | Controls false discovery rate |
| **None (but frame as exploratory)** | Honest exploration | Clearly label as exploratory, no strong claims |

- Pre-register which comparisons are primary (confirmatory) vs exploratory
- Primary comparisons get the correction; exploratory ones are reported without strong
  claims

**Common violations**:
- Testing many comparisons and reporting only the significant ones
- Not disclosing the total number of comparisons performed
- Applying corrections to exploratory analyses (misleading) or not applying to
  confirmatory ones (inflated false positives)

---

## 16. Reproducibility

**Principle**: Another researcher should be able to obtain the same results using only
the information you provide.

**Why it matters**: Science advances through building on verified results. If results
cannot be reproduced, they cannot be trusted or extended. Reproducibility failures are
a major crisis across scientific fields, including ML.

**Levels of reproducibility**:

| Level | Definition | Requirement |
|-------|-----------|-------------|
| **Computational** | Same code + same data → same numbers | Pinned environment, deterministic code, seeds |
| **Empirical** | Same method + same data → similar numbers | Clear method description, hyperparameters |
| **Conceptual** | Same idea + different implementation → similar conclusions | Robust findings, not implementation-dependent |

**How to apply**:
- Pin the full software environment (versions of all libraries)
- Use the [reproducibility-checklist.md](../templates/reproducibility-checklist.md)
- Document all hyperparameters including ones you did not tune
- Release code, data, and trained models when possible
- Verify reproducibility yourself before publication (re-run from scratch)

**Common violations**:
- Omitting "minor" implementation details that turn out to be critical
- Not pinning library versions (results change with updates)
- Not releasing code or data
- Reporting results from a codebase that has evolved since the experiments were run

**Key references**:
- Pineau, J., et al. (2021). "Improving Reproducibility in Machine Learning Research."
  *JMLR*, 22(164), 1-20.
- National Academies (2019). *Reproducibility and Replicability in Science*. National
  Academies Press.

---

## 17. Transparency and Complete Reporting

**Principle**: Report all results completely and honestly, including negative findings,
failed experiments, and limitations.

**Why it matters**: Selective reporting — publishing only positive results — creates a
biased literature where methods appear more effective than they are. Negative results
are just as informative as positive ones.

**How to apply**:
- Report results for ALL conditions, not just the ones that support the hypothesis
- Include failed experiments in the research log (they are valid milestones)
- Describe all limitations honestly
- Distinguish confirmatory from exploratory results (see [Experiment Execution](experiment-execution.md), Step 9)
- Use reporting frameworks appropriate to the study type (CONSORT, STROBE, PRISMA, ML
  Reproducibility Checklist)

**What to report**:
```
ALWAYS report:
- Results for all conditions (including baselines and failures)
- Variance across runs (not just the best run)
- Per-dataset breakdowns (not just averages)
- Negative and null results
- Deviations from the pre-registered protocol (labeled as exploratory)
- Hardware, runtime, and computational cost
- Known limitations and failure modes
```

**Common violations**:
- Reporting only the best model / best dataset / best metric
- Omitting runs that "didn't work" without explanation
- Presenting exploratory findings as if they were pre-registered
- Not disclosing computational cost

---

## 18. Ethical Design

**Principle**: Experiments must be designed with consideration for their broader impact
on people, communities, and the environment.

**Why it matters**: Research does not exist in a vacuum. Experiments involving human data,
large-scale compute, or potentially harmful applications carry ethical responsibilities
that must be addressed at the design stage, not as an afterthought.

**How to apply**:
- Human subjects: obtain IRB/ethics board approval before data collection
- Privacy: de-identify data, follow data protection regulations (GDPR, etc.)
- Dual use: assess whether the method could be misused; document mitigations
- Environmental cost: estimate and report computational carbon footprint for large-scale
  experiments
- Fairness: evaluate performance across demographic groups and document disparities
- Consent: ensure data was collected with appropriate consent for your use case

**Common violations**:
- Using human data without checking consent and licensing terms
- Not evaluating model fairness across demographic groups
- Ignoring environmental cost of large-scale training runs
- Not considering dual-use implications of the research

---

## 19. Quick-Reference Checklist

Use this checklist during experiment design to verify that all principles are addressed.
This complements (not replaces) the detailed protocol in [experiment-design.md](experiment-design.md).

```
EXPERIMENT DESIGN RIGOR CHECKLIST

Hypothesis & Falsifiability
[ ] Hypothesis is specific and falsifiable
[ ] Null hypothesis (H₀) is explicitly stated
[ ] Failure criteria defined ("I would abandon this if...")
[ ] Negative control included (condition where method should NOT work)

Controls & Comparisons
[ ] Strong, current baselines under identical conditions
[ ] Ablation controls isolate each component's contribution
[ ] Positive control validates the pipeline works
[ ] Equal resources (compute, tuning budget, data) across conditions

Randomization & Bias
[ ] Multiple random seeds (≥3, ideally 5+)
[ ] All seeds documented and different
[ ] Evaluation code written before seeing results
[ ] Qualitative analysis blinded to method identity (where applicable)

Pre-Registration
[ ] Primary metric defined with exact formula
[ ] Statistical test and decision rule pre-committed
[ ] Analysis plan locked before large-scale runs
[ ] Protocol committed to version control before results

Statistical Rigor
[ ] Sample size / number of runs justified
[ ] Appropriate statistical test selected
[ ] Multiple comparisons accounted for
[ ] Effect size reported alongside significance
[ ] Confidence intervals or variance reported

Validity
[ ] Internal: confounders identified and controlled
[ ] External: tested on multiple datasets or conditions
[ ] Construct: metrics actually measure the claimed construct
[ ] All variables operationalized with exact definitions

Reproducibility
[ ] Environment pinned (library versions, hardware documented)
[ ] All hyperparameters documented (including non-tuned ones)
[ ] Code, data, and models planned for release
[ ] Pipeline verified to be deterministic (same seed → same result)

Reporting
[ ] All conditions reported (including failures)
[ ] Variance across runs reported
[ ] Confirmatory vs exploratory clearly labeled
[ ] Limitations documented honestly

Ethics
[ ] Data licensing and consent verified
[ ] Privacy and de-identification addressed
[ ] Dual-use concerns assessed
[ ] Computational cost estimated
```

---

## 20. Recommended Reading

### Foundational Texts

- **Popper, K.** (1959). *The Logic of Scientific Discovery*. Routledge.
  — Falsifiability as the criterion of scientific hypotheses.

- **Fisher, R.A.** (1935). *The Design of Experiments*. Oliver & Boyd.
  — Foundational work on randomization, controls, and statistical testing.

- **Campbell, D.T. & Stanley, J.C.** (1963). *Experimental and Quasi-Experimental
  Designs for Research*. Houghton Mifflin.
  — Classic taxonomy of threats to internal and external validity.

- **Shadish, W.R., Cook, T.D., & Campbell, D.T.** (2002). *Experimental and
  Quasi-Experimental Designs for Generalized Causal Inference*. Houghton Mifflin.
  — Updated and expanded treatment of validity and causal inference in experiments.

- **Cohen, J.** (1988). *Statistical Power Analysis for the Behavioral Sciences*.
  2nd Edition. Lawrence Erlbaum.
  — Definitive reference on effect sizes and power analysis.

### Reproducibility and Open Science

- **National Academies of Sciences** (2019). *Reproducibility and Replicability in
  Science*. National Academies Press.
  — Comprehensive review of the reproducibility crisis and recommendations.

- **Nosek, B.A., et al.** (2018). "The preregistration revolution." *PNAS*, 115(11),
  2600-2606.
  — The case for pre-registration in empirical research.

- **Open Science Collaboration** (2015). "Estimating the reproducibility of psychological
  science." *Science*, 349(6251), aac4716.
  — Landmark study quantifying the reproducibility crisis.

### Machine Learning Specific

- **Pineau, J., et al.** (2021). "Improving Reproducibility in Machine Learning Research:
  A Report from the NeurIPS 2019 Reproducibility Program." *JMLR*, 22(164), 1-20.
  — ML-specific reproducibility guidelines and findings.

- **Dodge, J., et al.** (2019). "Show Your Work: Improved Reporting of Experimental
  Results." *EMNLP*.
  — Reporting standards for NLP experiments, expected validation performance.

- **Bouthillier, X., et al.** (2021). "Accounting for Variance in Machine Learning
  Benchmarks." *MLSys*.
  — How to properly measure and report variance in ML experiments.

- **Lipton, Z.C. & Steinhardt, J.** (2019). "Troubling Trends in Machine Learning
  Scholarship." *Queue*, 17(1).
  — Common methodological pitfalls in ML research.

- **Raji, I.D., et al.** (2021). "AI and the Everything in the Whole Wide World
  Benchmark." *NeurIPS Datasets and Benchmarks Track*.
  — Critique of construct validity in ML benchmarking.

- **Bergstra, J. & Bengio, Y.** (2012). "Random Search for Hyper-Parameter Optimization."
  *JMLR*, 13, 281-305.
  — Random search outperforms grid search for hyperparameter tuning.

### Statistical Methods

- **Wasserstein, R.L. & Lazar, N.A.** (2016). "The ASA Statement on p-Values." *The
  American Statistician*, 70(2), 129-133.
  — Official guidance on interpreting and reporting p-values.

- **Sullivan, G.M. & Feinn, R.** (2012). "Using Effect Size — or Why the P Value Is Not
  Enough." *Journal of Graduate Medical Education*, 4(3), 279-282.
  — Why effect sizes are essential alongside significance tests.

- **Button, K.S., et al.** (2013). "Power failure: why small sample size undermines the
  reliability of neuroscience." *Nature Reviews Neuroscience*, 14, 365-376.
  — The consequences of underpowered experiments.

### Ethics and Responsible Research

- **Gebru, T., et al.** (2021). "Datasheets for Datasets." *Communications of the ACM*,
  64(12), 86-92.
  — Framework for documenting datasets used in ML research.

- **Mitchell, M., et al.** (2019). "Model Cards for Model Reporting." *FAT*.
  — Framework for documenting ML models.

- **Strubell, E., Ganesh, A., & McCallum, A.** (2019). "Energy and Policy Considerations
  for Deep Learning in NLP." *ACL*.
  — Environmental impact of large-scale ML experiments.

---

*This reference is a companion to the [Experiment Design phase](experiment-design.md)
and the [Experiment Protocol template](../templates/experiment-protocol.md). Load it
during experiment design for guidance on applying these principles to your specific
experimental setup.*
