# Reading the evaluation code

Start with one spreadsheet row and follow these steps. A `test_case` is a
dictionary, not a separate class. Metrics are individual checks on that row.

| Step | File and function | Purpose |
| --- | --- | --- |
| 1 | `dataset_checks/dataset_loader.py`: `load_all_datasets` | Read Excel and build each test case. |
| 2 | `dataset_checks/preflight.py`: `run_preflight` | Check the supplied context before generating an answer. |
| 3 | `orchestrator/evaluation_orchestrator.py`: `run_test_case` | Coordinate generation, validation and scoring. |
| 4 | `prompts/prompt_builder.py`: `build_prompt` | Prepare the question and optional context. |
| 5 | `models/model_client.py`: `generate_response` | Ask the configured model for an answer. |
| 6 | `utils/response_validator.py`: `validate_response` | Detect generation errors or empty responses. |
| 7 | `evaluators/metric_manager.py`: `_evaluate` | Run suitable metrics in stage order. |
| 8 | `reporting/report_generator.py`: `generate_report` | Write the evaluation report. |

## How a metric is selected

`check_applicability` answers two questions in order:

1. Does this metric suit the row? `metric_registry.check_relevance` checks
   the evaluation rules, such as whether context was sent to the model.
2. Does it have the required data and settings? The metric module's
   `is_applicable` returns `(True, None)` or `(False, reason)`.

In `metrics/generic.py`, `_is_field_available` uses ordinary `if` statements:
expected output means `expected_answer`; both context fields use `context`.
Input and actual output are assumed to be supplied by the generation flow.
Tool and MCP data are currently unsupported.

## What the classes do

| Class | Responsibility |
| --- | --- |
| `GenericMetricModule` | Build registered DeepEval judge metrics using shared input checks and settings. |
| `ConfiguredMetricModule` | Build a judge metric that needs row-specific options. |
| `RubricModule` | Build a judge using written scoring instructions. |
| `OverlapModule` | Check whether a wording comparison can run. |
| `OverlapMetric` | Calculate ROUGE or BLEU wording overlap. |
| `NumericMatchMetric` | Check whether reference numbers appear in the answer. |
| `PatternMatchMetric` | Check the answer against an Excel regular expression. |
| `SemanticSimilarityMetric` | Compare answer and reference sentence embeddings. |

The common pattern is `is_applicable` (can it run?), `build` or
`build_for_case` (create it), then `measure` (calculate the result).

Generic modules are loaded only for names in the metric registry and preflight
list that do not already have dedicated modules. They must be single-turn judge
metrics accepting `model` and `threshold`, with a declared `_required_params`
list. Other metric types need a dedicated module. The application does not scan
every class in DeepEval.

## Reading results in the debugger

- `passed` / `failed`: the check ran and produced a verdict.
- `not_applicable`: required data is absent, the check is unsuitable, or an
  earlier gate stopped it. Read `reason` to distinguish these cases.
- `error`: the check could not complete; this is not a measured failure.
- `profile`: a code-generated category, not an Excel column.
- `context_role`: an estimate used when interpreting context-related scores.

A failed gate skips later gated stages. Safety checks can still run.
Human-review selection, run comparisons and calibration are separate from
the basic generate-and-score flow. Read those modules when investigating
their specific report fields.
