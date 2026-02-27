# Phase: Brainstorming & Ideation

## Goal

Generate a ranked set of candidate research directions using a structured
divergence-to-convergence loop that mitigates confirmation bias and premature commitment.

## Entry Conditions

- User has a broad topic area or a specific problem they want to explore
- OR: backtracking from Lit Review because the novelty gap was false

## Step-by-Step Protocol

### Step 1: Problem Landscape Map

Generate 10-30 candidate problems. For each, write a **one-sentence claim** — the
thing you would defend if the research succeeds.

**How to generate candidates:**
- Ask the user about their domain, constraints, and interests
- Search recent literature (arXiv, Semantic Scholar, Google Scholar) for active threads
- Identify gaps: what is assumed but unverified? what fails at scale? what lacks baselines?
- Check for "obvious baselines" that haven't been tried
- Look for cross-pollination: methods from domain A applied to domain B

**Output format:**
```
| # | One-sentence claim | Domain | Source of idea |
|---|-------------------|--------|----------------|
| 1 | "Fine-tuning on X improves Y by Z%" | NLP | Gap in [paper] |
| 2 | ... | | |
```

### Step 2: Trajectory Sketching

For the top ~10 candidates, outline 2-3 plausible research arcs:

```
Idea: [one-sentence claim]
Arc A: [baseline] → [method improvement] → [generalization test]
Arc B: [baseline] → [different method] → [ablation study]
Arc C: [theoretical analysis] → [empirical validation] → [limits characterization]
```

This step forces you to think about what the *whole project* looks like, not just the
first experiment.

### Step 3: Feasibility & Ethics Gate

Quickly eliminate directions that fail hard constraints:

**Feasibility filters:**
- Data: does the required data exist and is it accessible?
- Compute: is the compute budget realistic for the timeline?
- Skills: does the team have (or can quickly acquire) needed expertise?
- Time: can a meaningful result be obtained in the available window?

**Ethics filters:**
- Does it involve human subjects? → need IRB/ethics review path
- Does it use identifiable private data? → need consent/de-identification plan
- Could the output be misused? → need risk mitigation discussion
- Are there unsafe deployment contexts? → need safety constraints

Mark each candidate: PASS / CONDITIONAL (with mitigation) / FAIL.

### Step 4: Convergent Scoring

Score remaining candidates using the rubric from
[templates/scoring-rubric.md](../templates/scoring-rubric.md).

Use a 1-5 scale per criterion. Present results as a table:

```
| Idea | Feasible | Interesting | Novel | Ethical | Relevant | Evaluable | Reproducible | Robust | Risk-Ctrl | Mean |
|------|----------|-------------|-------|---------|----------|-----------|-------------|--------|-----------|------|
| #1   | 4        | 5           | 4     | 5       | 4        | 4         | 5           | 3      | 4         | 4.2  |
```

**Decision rule**: select ideas scoring ≥ 3.5 mean. If none qualify, return to Step 1
with refined scope.

### Step 5: Prototype & Falsify

For top 1-3 ideas, run **fast falsification** experiments:
- Implement the simplest possible version (hours, not days)
- Run a "sanity check" baseline — if the baseline already solves it, the idea is moot
- Check for data leakage and label quality issues
- Test on a tiny subset first

**Key question**: "What is the cheapest experiment that would make me abandon this idea?"

If the idea survives falsification, proceed. If not, log the result (negative findings
are valid milestones!) and return to the scoring table.

### Step 6: Commit to Protocol

Once a direction survives:
1. Write a 1-paragraph research statement (claim + scope + method sketch)
2. Identify the primary metric, primary dataset, and primary baseline
3. Lock these before scaling up — this is your informal "preregistration"
4. Log the commitment in LOGBOX

### Step 7: Initialize Exploration Directory

If the project uses the exploration structure (see SKILL.md § File Management):

1. Create `explorations/NNN-slug/` — next sequential number + kebab-case name
2. Save brainstorming artifacts to the exploration directory:
   - `brainstorm.md` — scoring table, research statement, and falsification results
3. Add a row to the Exploration Registry in LOGBOX (status: `active`, parent: `—` or
   the exploration this was forked from)

**Multiple viable ideas**: if brainstorming produces 2-3 ideas above the 3.5 threshold,
create an exploration directory for each. Set the one the user chooses to pursue first
as `active` and the others as `paused`.

**Forking from a failed exploration**: when pivoting after a failed direction, set the old
exploration to `archived` and create the new one with the old ID as `parent`. Promote any
reusable artifacts (evidence maps, datasets) to `shared/`.

## Prompt Bank

Use these questions to push thinking during brainstorming:

**Problem & contribution:**
- What is the smallest, sharpest claim we want to defend?
- If we succeed, what changes for (a) theory, (b) practice, (c) measurement?
- What is the "obvious baseline" that must be beaten to matter?

**Mechanism & assumptions:**
- What assumption is currently taken for granted that may be false?
- What mechanism would explain the effect if the result is real?
- What conditions would make the method fail?

**Data & measurement:**
- What is the target population/distribution? What is the sampling process?
- What labels are assumed "ground truth," and how might they be biased?
- What data leakage pathways exist?

**Evaluation & falsification:**
- What metric best matches the real objective (and what metric is easiest to game)?
- What "sanity check" would immediately invalidate our pipeline if it fails?
- What ablations distinguish "real contribution" from incidental engineering?

## Exit Criteria

- [ ] At least one idea scores ≥ 3.5/5 on the scoring rubric
- [ ] Fast falsification did not kill the top pick
- [ ] Primary metric, dataset, and baseline are identified (not yet locked)
- [ ] Research statement (1 paragraph) is written
- [ ] Exploration directory created (if using multi-exploration structure)
- [ ] LOGBOX entry recorded

## Transition

**Forward → Lit Review**: carry the research statement, initial candidate papers, and
scoring table.

**Backward ← Lit Review**: if novelty gap turns out to be false, archive the current
exploration (`archived` in LOGBOX), return here, and either re-score or create a new
exploration.

**Fork → New Exploration**: if multiple viable ideas exist, create separate explorations
and let the user choose which to activate first.
