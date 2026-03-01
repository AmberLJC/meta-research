# Phase: Experiment Design

## Goal

Create a locked, reviewable experimental protocol for a specific approved hypothesis.
Every hypothesis must be verified by experiment — no claim is accepted without empirical
evidence from a rigorous test. The protocol specifies exactly what will be tested, how,
and what constitutes success, before running large-scale experiments.

## Entry Conditions

- At least one hypothesis has `status: approved` in the research tree (from Judgment)
- Evidence map with identified baselines, datasets, and methods (from Literature Survey)
- Hypothesis statement is specific and falsifiable

## Principles of Rigorous Experiment Design

These principles must guide every protocol. Violations of any principle weaken the
experiment's ability to verify the hypothesis.

### Falsifiability & Controls

| Principle | Key rule | Watch for |
|-----------|----------|-----------|
| **Falsifiability** | State what would disprove H before designing the test. Define H₀ explicitly. | Experiments where every outcome "confirms"; ceiling-effect tasks; post-hoc reinterpretation |
| **Controls** | Every claim needs a control that eliminates the most plausible alternative explanation. | Weak/outdated baselines; unequal tuning budgets; missing ablations or negative controls |
| **Minimal sufficiency** | Simplest experiment that answers the question. Escalate effort only when simpler tests are inconclusive. | Running massive benchmarks before a pilot; confounding multiple hypotheses in one experiment |

**Control types**: baseline (strong existing method, identical conditions), ablation (method minus one component), negative (method should NOT work), positive (method should obviously work), placebo (identical procedure, no active ingredient).

**Effort hierarchy**: thought experiment → back-of-envelope → pilot study → focused experiment → large-scale study.

### Randomization & Bias Reduction

| Principle | Key rule | Watch for |
|-----------|----------|-----------|
| **Randomization** | Random seeds for splits, init, training order. Multiple seeds per condition (≥3, ideally 5+). Document all seeds. | Single seed; cherry-picked best seed; non-random temporal splits |
| **Blinding** | Write evaluation code before seeing results. Anonymize method identity in qualitative analysis. | Iteratively tuning your method while baselines stay at defaults; adjusting criteria after results |
| **Pre-registration** | Lock the analysis plan before large-scale runs. Commit protocol to git before results. Any post-hoc analysis = EXPLORATORY. | Outcome switching; HARKing; undisclosed post-hoc comparisons |

### Statistical Rigor

| Principle | Key rule | Watch for |
|-----------|----------|-----------|
| **Power** | Estimate minimum meaningful effect size. Ensure enough runs/samples to detect it (power ≥ 0.80). | Underpowered experiments claiming "no effect"; optional stopping |
| **Effect size** | Report magnitude (e.g., "3.2 pts, 95% CI [1.8, 4.6]"), not just p-values. Contextualize vs. human perf / inter-method gap. | "Significant" without magnitude; trivially small improvements |
| **Multiple comparisons** | Count total comparisons (N×M×K). Apply Bonferroni/Holm/BH for confirmatory; label rest exploratory. | Reporting only significant comparisons; undisclosed total count |
| **Independence** | Identify unit of analysis. Use paired/clustered tests when observations are correlated. | Treating per-token metrics as independent; ignoring document-level clustering |

### Validity

| Principle | Key rule | Watch for |
|-----------|----------|-----------|
| **Internal** | Isolate the independent variable. Match conditions on compute, data, tuning budget. | Data leakage; hyperparameter bias; compute confounds; selection bias |
| **External** | Test on multiple datasets/domains. Report per-dataset results, not just averages. | Single-benchmark claims of generality; hidden per-dataset failures |
| **Construct** | Ask: "Does this metric actually measure what I claim?" Use behavioral tests to probe the underlying capability. | Equating benchmark accuracy with "understanding"; shortcut learning |
| **Operationalization** | Define every metric with exact formula, averaging policy, edge cases, and evaluation conditions before running. | Vague metric names; inconsistent edge-case handling across methods |

### Reproducibility & Transparency

| Principle | Key rule | Watch for |
|-----------|----------|-----------|
| **Reproducibility** | Pin environment. Document all hyperparameters. Release code/data/models. Verify determinism (same seed → same result). | Omitted implementation details; unpinned library versions; unreleased code |
| **Transparency** | Report ALL conditions including failures. Distinguish confirmatory vs exploratory. Include negative results. | Best-run-only reporting; omitted failed experiments; undisclosed compute cost |
| **Ethics** | Check data consent/licensing. Assess dual-use. Estimate compute carbon footprint. Evaluate fairness across groups. | Using data without consent verification; ignoring environmental cost |

## Step-by-Step Protocol

### Step 1: Select the Hypothesis

Choose the highest-priority approved hypothesis from the research tree. Record:
- Hypothesis ID (e.g., H1)
- Full statement
- Motivation
- Relevant findings from the literature survey

The hypothesis statement from the research tree IS the testable claim — no further
sharpening is needed (that was done during Hypothesis Generation and Judgment).

### Step 2: Define Variables and Controls

**Independent variables** (what you manipulate):
- List each factor and its levels (e.g., model size: {small, medium, large})

**Dependent variables** (what you measure):
- Primary metric: [one metric that is the headline result]
- Secondary metrics: [clearly labeled as secondary/exploratory]
- Metric definitions: [exact formula, averaging policy, edge cases]

**Controls:**
- **Baseline controls**: strong existing methods run under identical conditions
- **Ablation controls**: your method minus one component at a time
- **Negative controls**: conditions where your method SHOULD NOT work (sanity check)
- **Positive controls**: conditions where improvement is expected to be obvious

**Confounders to address:**
- Hyperparameter tuning budget (same budget for baselines and your method)
- Data leakage (train/test overlap, temporal leakage, test contamination)
- Compute differences (compare at equal compute, not just equal epochs)

### Step 3: Data Plan

```
DATA PLAN
- Dataset(s): [name, source, version, access method]
- License/terms: [what is allowed]
- Population/coverage: [what does this data represent? what is missing?]
- Size: [train/val/test counts]
- Split strategy: [random / stratified / temporal / predefined]
- Leakage prevention: [how you ensure no contamination]
- Preprocessing: [deterministic steps, versioned scripts]
- Dataset documentation: [Datasheet or Data Statement — fill if new dataset]
```

**Critical**: define splits BEFORE looking at test data. If using an existing benchmark,
use the canonical splits. If creating new splits, document the procedure and random seed.

### Step 4: Training and Compute Plan (for ML)

```
COMPUTE PLAN
- Model architecture: [family, size, key design choices]
- Hyperparameters:
  - Fixed (not tuned): [list with values and justification]
  - Tuned: [list with search space, search method, budget]
  - Tuning done on: [validation set ONLY — never test]
- Random seeds: [number of runs per condition, e.g., 5]
- Hardware: [GPU/TPU type, count, memory]
- Expected runtime: [per run, total]
- Checkpointing: [frequency, what is saved]
- Early stopping: [criterion, patience]
```

### Step 5: Analysis Plan (Pre-Commit)

This is the most important step for rigor — it is your informal preregistration.

```
ANALYSIS PLAN
Primary analysis:
- Compare [method] vs [baselines] on [primary metric]
- Statistical test or comparison method: [e.g., paired bootstrap, Wilcoxon]
- Decision rule: [what constitutes "better"? just point estimate? CI must exclude 0?]

Uncertainty reporting:
- Across seeds: [mean +/- std, or CI from N runs]
- Across datasets: [if applicable]
- Visualization: [box plots, violin plots, or distribution histograms]

Multiple comparisons:
- How many comparisons: [N models x M datasets x K metrics]
- Correction: [Bonferroni / Holm / none-but-frame-as-exploratory]

Ablation plan:
- Component 1 removed: expected effect = [...]
- Component 2 removed: expected effect = [...]
- Component N removed: expected effect = [...]

Error analysis plan:
- Slices/subgroups to check: [e.g., by language, by difficulty, by domain]
- Qualitative audit: [N random examples from errors, categorized]
- Behavioral tests: [specific input patterns to probe, CheckList-style]

Exploratory analyses (clearly labeled):
- [anything you want to look at but did not commit to in advance]
```

### Step 6: Reproducibility Artifacts Plan

```
ARTIFACT PLAN
- Code repository: [URL, structure, entry points]
- Environment: [conda yml / pip lockfile / Docker container]
- Experiment tracking: [tool: MLflow / W&B / DVC / etc.]
- Run naming scheme: [e.g., {method}_{dataset}_{seed}_{timestamp}]
- How to map paper tables -> run IDs: [documented mapping]
- Data release: [plan or justification for not releasing]
- Model release: [plan or justification for not releasing]
- Expected storage footprint: [for artifacts, checkpoints, logs]
```

### Step 7: Ethics and Risk Review

```
ETHICS REVIEW
- Human subjects: [yes/no — if yes, IRB/ethics review status]
- Identifiable data: [yes/no — if yes, de-identification plan]
- Dual-use concerns: [could this be misused? mitigation plan]
- Environmental cost: [estimated compute carbon footprint if large-scale]
- Model documentation: [Model Card planned if releasing a model]
```

### Step 8: Lock the Protocol

Once all sections are filled:
1. Write the full protocol using [templates/experiment-protocol.md](../templates/experiment-protocol.md)
2. Save it to the hypothesis experiment directory: `experiments/H[N]-slug/docs/protocol.md`
3. Have a collaborator review it (or self-review after 24h)
4. Update the research tree: `experiment.status: locked`
5. Log the protocol lock in the research log
6. Any deviations from this point forward must be logged as EXPLORATORY

### Per-Hypothesis Directory Structure

Each hypothesis gets a self-contained directory with standardized subdirectories:

```
experiments/H[N]-slug/
├── docs/                     # Hypothesis-level documentation
│   ├── protocol.md           # Locked experiment protocol
│   ├── analysis.md           # Consolidated analysis (sanity, primary, ablation, error)
│   └── notes.md              # Informal observations, ideas, debugging notes
├── src/                      # Experiment code
│   ├── train.py              # Training scripts
│   ├── evaluate.py           # Evaluation scripts
│   ├── analyze.py            # Analysis and visualization scripts
│   └── configs/              # Hyperparameter configs, model configs
├── data/                     # Hypothesis-specific data (symlink shared data)
│   ├── raw/                  # Raw data or symlinks to shared/data/
│   ├── processed/            # Preprocessed, split-ready data
│   └── splits/               # Train/val/test split definitions (indices or files)
├── results/                  # Outputs from experiment runs
│   ├── runs/                 # Per-run outputs: {method}_{dataset}_{seed}/
│   ├── tables/               # Aggregated result tables (CSV/JSON)
│   └── figures/              # Generated plots and visualizations
└── logs/                     # Execution and tracking logs
    ├── runs/                 # Per-run training logs (stdout, metrics per step)
    ├── tracking/             # Experiment tracker exports (MLflow/W&B dumps)
    └── sanity-checks.md      # Sanity check results before analysis
```

**Shared resources** live at the project level to avoid duplication:

```
project/
├── shared/
│   └── data/                 # Datasets used by multiple hypotheses (downloaded once)
├── experiments/
│   ├── H1-slug/              # Full per-hypothesis directory (structure above)
│   ├── H2-slug/
│   └── ...
```

**Rules:**
- Symlink shared datasets into `data/raw/` rather than copying
- Run naming: `{method}_{dataset}_{seed}_{timestamp}` inside `results/runs/`
- `docs/protocol.md` is the locked protocol — never modify after locking
- Everything in `logs/` is append-only during execution

## Rigor Checklist

Verify before locking the protocol:

```
EXPERIMENT DESIGN RIGOR CHECKLIST

Falsifiability & Controls
[ ] Hypothesis is specific and falsifiable; H₀ stated explicitly
[ ] Failure criteria defined ("I would abandon this if...")
[ ] Strong baselines under identical conditions; ablations isolate each component
[ ] Negative control included; positive control validates pipeline

Randomization & Bias
[ ] Multiple random seeds (≥3, ideally 5+), all documented
[ ] Evaluation code written before seeing results
[ ] Analysis plan locked before large-scale runs; protocol in version control before results

Statistical Rigor
[ ] Sample size / number of runs justified for expected effect size
[ ] Statistical test and decision rule pre-committed
[ ] Multiple comparisons accounted for; effect size reported alongside significance

Validity
[ ] Internal: confounders identified and controlled
[ ] External: tested on multiple datasets or conditions
[ ] Construct: metrics actually measure the claimed construct
[ ] All variables operationalized with exact definitions

Reproducibility & Transparency
[ ] Environment pinned; all hyperparameters documented
[ ] Code, data, and models planned for release
[ ] All conditions reported (including failures); confirmatory vs exploratory labeled

Ethics
[ ] Data licensing and consent verified
[ ] Dual-use concerns assessed; computational cost estimated
```

## Exit Criteria

- [ ] Hypothesis ID is recorded and linked to the research tree
- [ ] Variables, controls, and confounders documented
- [ ] Data plan with splits and leakage prevention
- [ ] Compute plan with seed count and hardware specs
- [ ] Analysis plan pre-committed (primary and ablation)
- [ ] Artifact plan with environment and tracking setup
- [ ] Ethics review completed
- [ ] Protocol locked; research tree updated (`experiment.status: locked`)
- [ ] Per-hypothesis directory created with all subdirectories
- [ ] Research log entry recorded

## Transition

**Forward → Experiment Execution**: carry the locked protocol. The hypothesis is ready
for implementation and execution.

**Backward → Literature Survey**: if design reveals that a critical baseline or dataset
is missing from the evidence map, return to find it.

**Backward ← Experiment Execution**: if pipeline bugs or data leakage are found during
execution, return here to fix the protocol and re-run.

**Backward ← Reflection**: if reflection identifies missing experiments, return here to
design additional tests for new or refined hypotheses.

## Recommended Reading

**Foundational**: Popper (1959) *Logic of Scientific Discovery*; Fisher (1935) *Design of Experiments*; Campbell & Stanley (1963) *Experimental and Quasi-Experimental Designs*; Shadish, Cook & Campbell (2002) *Generalized Causal Inference*; Cohen (1988) *Statistical Power Analysis*.

**Reproducibility & open science**: National Academies (2019) *Reproducibility and Replicability in Science*; Nosek et al. (2018) "The preregistration revolution" PNAS; Open Science Collaboration (2015) Science.

**ML-specific**: Pineau et al. (2021) "Improving Reproducibility in ML Research" JMLR; Dodge et al. (2019) "Show Your Work" EMNLP; Bouthillier et al. (2021) "Accounting for Variance in ML Benchmarks" MLSys; Lipton & Steinhardt (2019) "Troubling Trends in ML Scholarship"; Raji et al. (2021) "AI and the Everything Benchmark" NeurIPS.

**Statistics**: Wasserstein & Lazar (2016) "ASA Statement on p-Values"; Sullivan & Feinn (2012) "Using Effect Size"; Button et al. (2013) "Power failure" Nature Rev Neuroscience.

**Ethics**: Gebru et al. (2021) "Datasheets for Datasets" CACM; Mitchell et al. (2019) "Model Cards" FAT*; Strubell et al. (2019) "Energy and Policy Considerations" ACL.
