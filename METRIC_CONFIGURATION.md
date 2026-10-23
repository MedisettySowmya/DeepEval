# Additional QA metrics

## Configuration loading

`config/settings.py` contains all suite settings and startup validation. Model settings and
scalar runtime options accept environment overrides, with precedence:
process environment > project-root `.env` > Python defaults. Copy `.env.example`
to `.env` to customize local model settings. `.env` is ignored by Git.
Values are read when the module imports; restart a debug session after edits.
Blank values are errors. Boolean values accept true/false, yes/no, or 1/0.
The `.env` reader does not expand `${...}` or alter process environment variables.
Both model roles currently use Ollama. No LiteLLM or Anthropic integration is
included. Changing an Ollama model requires changing its configured model name;
the model must already be available on the configured Ollama server.

Startup validation rejects unsupported providers, invalid HTTP endpoints,
nonpositive timeouts, invalid numeric ranges, and unknown instruction variants.
Runtime prompt overrides also reject unknown variants. Validation checks local
configuration only; it does not test model-server availability.
Metric policies and per-metric thresholds are also in `config/settings.py`.

## Model integration

`models/model_client.py` has three functions:

- `generate_response(prompt)` sends the test prompt to the generator using the
  Ollama SDK and returns its answer. SDK errors reach the response validator unchanged.
- `create_judge()` constructs the DeepEval Ollama wrapper. DeepEval metrics call
  that wrapper later to score answers; construction does not verify server availability.
- `uses_different_models()` compares configured provider and model names for reports.
  Different names do not prove independent judgment or different underlying weights.

Each role has its own model, endpoint, temperature, and timeout. `GENERATOR_TIMEOUT`
and the new `JUDGE_TIMEOUT` default to 300 seconds and configure HTTP client timeouts;
they are not a time limit for an entire evaluation or its retries.

Execution flow: settings load and validate, the orchestrator generates an answer,
then metrics use the judge to evaluate it. Reports record the configured identities.
The settings module makes no model requests.

The former `model_config.py`, `generator_resolver.py`, `judge_model_resolver.py`,
and `ollama_config.py` have been replaced. Project imports now use
`from config import settings` and functions from `models.model_client`.
External scripts using the old imports must migrate too. The UI config response
now uses `different_models` instead of `judge_independent`.

Offline configuration and client checks:
`python -m pytest tests/test_config_settings.py tests/test_model_client.py -o addopts= -q`

The answer pipeline has 20 metrics. Pytest also runs 3 separate contextual
dataset checks: 23 connected metrics in total. JSON correctness is excluded.

AnswerCompleteness is a GEval coverage rubric using Ground_Truth; it runs only
with a comparable reference. CitationCorrectness and InstructionPriority are
custom GEval rubrics, not built-in DeepEval metrics. They run in the ungated final
stage and remain advisory until calibrated.

Enable citation checks only on cases that request citations and supply labeled
sources in Context (for example `[1] Title: supporting passage`). Add an optional
Excel `Metric_Options` column containing:

```json
{"CitationCorrectness":{"enabled":true,"sources_labeled":true}}
```

For injection tests, place the attack in Context and enable:

```json
{"InstructionPriority":{"enabled":true}}
```

The trusted instruction is taken from the actual assembled prompt, not from an
invented scoring policy. Choose an instruction variant in settings whose
text clearly defines how to handle untrusted context. No instruction sent means
this metric skips. Sources/context must actually have been sent to the model.
These checks make additional judge calls; they do not prove factual truth.

Programmatic cases may supply the same dictionary as `metric_options`.
Suite-wide defaults may be added to `settings.METRIC_OPTIONS`, but per-row
configuration is preferable for mixed datasets. Calibration snapshots must
preserve the sent prompt and actual trusted instructions in metric options.

ROUGELMetric and BLEUMetric are local lexical metrics using lowercase word and
punctuation tokens, a single reference, and scores in [0,1]. ROUGE-L is LCS F1.
BLEU uses clipped n-gram precision, effective order up to 4, brevity penalty and
no smoothing (a zero precision yields zero). These are sentence-level variants,
not SacreBLEU corpus scores. Both require a comparable reference and remain
advisory even when their score is below threshold. They never fail pytest on
their own. No extra dependencies are needed.

Offline checks:
`python -m pytest tests/test_metrics.py tests/test_preflight_connection.py -o addopts= -q`
