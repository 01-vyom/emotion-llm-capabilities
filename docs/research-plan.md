# Research Plan

Date context: 2026-05-24 project snapshot.

## Question

Can affective framing or affect-like internal representations improve language
model performance on boring tasks where failures are usually quiet, sparse, and
low-salience?

The target is not subjective emotion. The operational target is a behavioral
control signal: does the intervention change persistence, checking, omission
rate, schema compliance, shortcutting, or safety behavior?

## Motivation

Most public LLM evaluation rewards salient reasoning. Many deployed tasks are
not like that. They are repetitive and exacting:

- classify many similar messages
- extract every field from forms or contracts
- check every row of a table
- preserve schema and source fidelity
- review compliance clauses
- do repetitive code maintenance

These tasks fail through omissions, false positives, format-valid wrong answers,
position-wise decay, and rare missed details.

## Stage 1: Prompt-Only Pilot

Stage 1 uses one open model and one objective benchmark.

- Model: `Qwen/Qwen3-8B`
- Dataset: `PolyAI/banking77`
- Main task: classify each user message into exactly one intent label
- Engine: `vllm` for A100 runs; `transformers` for fallback runs
- Decoding: deterministic, `temperature=0`
- Primary output: label only, no rationale

Conditions:

- `neutral_empty`
- `neutral_length_matched`
- `positive_pride`
- `negative_failure_avoidance`
- `high_stakes_audit`
- `high_stakes_extreme`
- `calm_meticulous`
- `boredom_antidote_repetitive`

Primary metrics:

- accuracy
- macro-F1
- parse rate
- bootstrap confidence interval for accuracy
- confusion counts

Decision rule:

An affect cue is interesting only if it beats both the empty neutral baseline
and the length-matched neutral control. If it does not, the result is still
useful: the active ingredient is likely instruction strength, prompt length, or
task salience rather than emotion.

## Stage 2: Broader Tedious-Task Suite

Add task families with objective or near-objective scoring:

- IFEval for instruction compliance
- RULER or NoLiMa subset for low-salience long-context retrieval
- CUAD for contract clause extraction
- TAT-QA or FeTaQA for table QA
- SpreadsheetBench subset for office-style spreadsheet work
- CodeXGLUE code refinement for repetitive code repair

Use two or three models:

- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-14B`
- `meta-llama/Llama-3.1-8B-Instruct` or `google/gemma-3-12b-it`

## Stage 3: Activation-Level Study

The Anthropic emotion paper induced and found emotion concepts by generating
synthetic stories where characters experienced target emotions, then extracting
linear directions from residual-stream activations. This project mirrors that
idea in open models.

Planned contrastive concepts:

- calm vs desperate
- focused vs distracted
- patient vs rushed
- responsible vs careless
- curious vs bored

Procedure:

1. Generate synthetic affect vignettes with matched topics.
2. Collect hidden states at selected residual-stream layers.
3. Compute difference-in-means vectors for each concept pair.
4. Validate probes on held-out vignettes without obvious keyword leakage.
5. Steer neutral task prompts with those vectors.
6. Test dose response, layer sensitivity, and opposite-sign controls.

Key causal tests:

- neutral prompt with affect vector
- affective prompt with opposite vector
- random vector control
- shuffled-label vector control
- prompt x vector crossed design
- safety and shortcutting audit

## Stage 4: Optional Adapter Pilot

If prompt and activation steering show a repeatable signal, train a small QLoRA
adapter such as `calm_verifier` or `patient_auditor`.

This is optional. The stronger first article is prompt plus activation steering,
not fine-tuning.

## Statistical Plan

Use paired evaluation: every item is run under every condition.

Report:

- absolute percentage-point differences
- paired deltas against neutral
- cluster bootstrap confidence intervals
- prompt-condition variance
- model x condition interactions

Avoid best-of-many prompt claims. Select prompts on a dev split and report
locked held-out results.

## Safety Plan

Every condition that improves task performance must be paired with safety and
reliability checks:

- hallucinated field rate
- refusal rate
- unsafe compliance
- sycophancy
- shortcutting or reward-hacking behavior
- token cost and verbosity
- format-valid but wrong outputs

Extreme stress is a safety stress test, not a deployment recommendation.
