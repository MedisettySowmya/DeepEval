import html
from datetime import datetime

from config import settings

from models.model_client import uses_different_models

_STYLE = """
:root {
  --bg: #f4f6f8; --surface: #fff; --surface-2: #eef1f5; --line: #d7dde5;
  --fg: #1b2430; --fg-dim: #5c6878; --fg-faint: #8a95a3;
  --accent: #0d7d8c; --accent-soft: #e0f2f4;
  --ok: #1d7a4c; --ok-soft: #e2f3e9;
  --bad: #a52f2f; --bad-soft: #fbe6e6;
  --warn: #96600a; --warn-soft: #fdf0d8;
  --human: #6b3fa0; --human-soft: #efe7f9;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #11161c; --surface: #1a2129; --surface-2: #222b35; --line: #313c48;
    --fg: #e6ebf1; --fg-dim: #9aa6b4; --fg-faint: #6e7986;
    --accent: #45b8c6; --accent-soft: #12333a;
    --ok: #5fcc90; --ok-soft: #13301f;
    --bad: #f08a8a; --bad-soft: #3a1b1b;
    --warn: #e0ad53; --warn-soft: #332611;
    --human: #b996e8; --human-soft: #261a38;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg); line-height: 1.55;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; font-size: 15px;
}
.wrap { max-width: 1100px; margin: 0 auto; padding-inline: 16px; padding-block: 28px 56px; }
h1 { font-size: 26px; font-weight: 600; margin: 0 0 4px; }
.sub { color: var(--fg-dim); margin: 0 0 22px; font-size: 13.5px; }
h2 {
  font-size: 11px; text-transform: uppercase; letter-spacing: .09em;
  color: var(--fg-faint); font-weight: 600; margin: 28px 0 12px;
}
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; }
.verdict { font-size: 17px; font-weight: 600; margin: 0 0 6px; }
.verdict.pass { color: var(--ok); }
.verdict.fail { color: var(--bad); }
.verdict-note { margin: 0; color: var(--fg-dim); font-size: 13.5px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 9px; padding: 12px 14px; }
.tile .n { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .l { font-size: 11.5px; color: var(--fg-faint); text-transform: uppercase; letter-spacing: .06em; }
.kv { display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: 13px; }
.kv div { min-width: 0; }
.kv .k { color: var(--fg-faint); }
.kv .v { font-family: ui-monospace, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
.case { background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 16px 18px; margin-bottom: 12px; }
.case-head { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 12px; }
.case-head .q { font-weight: 600; min-width: 0; overflow-wrap: anywhere; }
.tag {
  font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11.5px;
  background: var(--accent-soft); color: var(--accent); border-radius: 4px; padding: 1px 6px; flex: none;
}
.block { margin-top: 10px; background: var(--surface-2); border-radius: 7px; padding: 10px 12px; font-size: 13px; }
.block .bl { font-size: 11px; text-transform: uppercase; letter-spacing: .07em; color: var(--fg-faint); margin-bottom: 3px; }
.block .bv { font-family: ui-monospace, Menlo, Consolas, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
.tablewrap { margin-top: 12px; overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; min-width: 520px; }
th {
  text-align: left; font-size: 10.5px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--fg-faint); font-weight: 600; padding: 0 10px 6px 0; border-bottom: 1px solid var(--line);
}
td { padding: 6px 10px 6px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
td.m { font-family: ui-monospace, Menlo, Consolas, monospace; }
td.num { font-variant-numeric: tabular-nums; }
td.why { color: var(--fg-dim); }
tr.stage td {
  font-size: 11px; text-transform: uppercase; letter-spacing: .07em;
  color: var(--fg-faint); padding-top: 12px; border-bottom: none;
}
.pill { display: inline-block; font-size: 11px; font-weight: 500; padding: 1px 7px; border-radius: 99px; white-space: nowrap; }
.p-passed { background: var(--ok-soft); color: var(--ok); }
.p-failed { background: var(--bad-soft); color: var(--bad); }
.p-error { background: var(--bad-soft); color: var(--bad); }
.p-not_applicable { background: var(--surface-2); color: var(--fg-faint); }
.p-trusted { background: var(--ok-soft); color: var(--ok); }
.p-advisory { background: var(--warn-soft); color: var(--warn); }
.p-requires_human_review { background: var(--human-soft); color: var(--human); }
.flag { margin-top: 10px; padding: 9px 11px; border-radius: 7px; font-size: 12.5px; background: var(--human-soft); color: var(--human); }
.empty { color: var(--fg-dim); }
footer { margin-top: 34px; padding-top: 14px; border-top: 1px solid var(--line); font-size: 12.5px; color: var(--fg-faint); }
"""

_STAGE_NAMES = {
    0: "Stage 0 — generation",
    1: "Stage 1 — deterministic (no AI judge)",
    2: "Stage 2 — relevance",
    3: "Stage 3 — grounding (answer vs context)",
    4: "Stage 4 — reference (answer vs ground truth)",
    5: "Stage 5 — safety",
}

def _esc(value):
    return html.escape("" if value is None else str(value))

def _pill(value):
    text = "—" if value in (None, "") else str(value)
    return f'<span class="pill p-{_esc(value)}">{_esc(text.replace("_", " "))}</span>'

def _context_to_text(context):
    if not context:
        return ""
    if isinstance(context, list):
        return "\n".join(str(item) for item in context)
    return str(context)

def _score(value):
    if value is None:
        return "—"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return _esc(value)

def _collect_counts(test_case_results):
    """Everything the summary needs, counted in one pass. Pure reading --
    no metric is executed to produce these numbers."""

    counts = {
        "cases": len(test_case_results),
        "classification": {"VERIFIED": 0, "EVALUATED": 0, "UNVERIFIABLE": 0},
        "status": {"passed": 0, "failed": 0, "error": 0, "not_applicable": 0},
        "gating": {"trusted": 0, "advisory": 0, "requires_human_review": 0, "not_applicable": 0},
        "blocking": [],
        "needs_human": 0,
    }

    for item in test_case_results:
        result = item["result"]

        classification = (result.get("classification") or {}).get("classification")
        if classification in counts["classification"]:
            counts["classification"][classification] += 1

        if result.get("human_review"):
            counts["needs_human"] += 1

        for metric_key, metric_result in (result.get("evaluation") or {}).items():

            status = metric_result.get("status")
            if status in counts["status"]:
                counts["status"][status] += 1

            gating = metric_result.get("gating_status")
            if gating in counts["gating"]:
                counts["gating"][gating] += 1

            if gating == "trusted" and status in ("failed", "error"):
                counts["blocking"].append({
                    "input": item["test_case"].get("input"),
                    "metric": metric_key,
                    "status": status,
                    "score": metric_result.get("score"),
                })

    return counts

def _metric_rows(evaluation, metric_diagnoses):
    """Metric table for one test case, grouped by the stage each metric ran
    in so the ladder is visible rather than an alphabetical jumble."""

    if not evaluation:
        return '<p class="empty">No metrics ran for this test case.</p>'

    ordered = sorted(
        evaluation.items(),
        key=lambda pair: (
            pair[1].get("stage") if pair[1].get("stage") is not None else 99,
            pair[0],
        ),
    )

    rows = []
    current_stage = object()

    for metric_key, metric_result in ordered:

        stage = metric_result.get("stage")
        if stage != current_stage:
            current_stage = stage
            label = _STAGE_NAMES.get(stage, "Not staged")
            rows.append(f'<tr class="stage"><td colspan="6">{_esc(label)}</td></tr>')

        diagnosis = (metric_diagnoses or {}).get(metric_key) or {}

        why = metric_result.get("interpretation") or metric_result.get("reason") or ""
        if diagnosis.get("recommended_action"):
            why = f"{why} → {diagnosis['recommended_action']}" if why else diagnosis["recommended_action"]

        rows.append(
            "<tr>"
            f'<td class="m">{_esc(metric_key)}</td>'
            f"<td>{_pill(metric_result.get('status'))}</td>"
            f'<td class="num">{_score(metric_result.get("score"))}</td>'
            f'<td class="m">{_esc(metric_result.get("tier"))}</td>'
            f"<td>{_pill(metric_result.get('gating_status'))}</td>"
            f'<td class="why">{_esc(why)}</td>'
            "</tr>"
        )

    return (
        '<div class="tablewrap"><table><thead><tr>'
        "<th>Metric</th><th>Result</th><th>Score</th><th>Tier</th>"
        "<th>Counts toward pass/fail?</th><th>Reason</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )

def _case_section(item):

    test_case = item["test_case"]
    result = item["result"]

    validation = result.get("validation") or {}
    classification = result.get("classification") or {}
    root_cause = result.get("root_cause_analysis") or {}

    parts = [
        '<div class="case">',
        '<div class="case-head">',
        f'<span class="q">{_esc(test_case.get("input"))}</span>',
        f'<span class="tag">{_esc(test_case.get("category"))}</span>',
    ]

    if result.get("profile"):
        parts.append(f'<span class="tag">{_esc(result.get("profile"))}</span>')

    parts.append(_pill(classification.get("classification")))
    parts.append("</div>")

    context_text = _context_to_text(test_case.get("context"))
    if context_text:
        parts.append(
            '<div class="block"><div class="bl">Context given</div>'
            f'<div class="bv">{_esc(context_text)}</div></div>'
        )

    if test_case.get("expected_answer"):
        parts.append(
            '<div class="block"><div class="bl">Ground truth</div>'
            f'<div class="bv">{_esc(test_case.get("expected_answer"))}</div></div>'
        )

    if validation.get("valid"):
        parts.append(
            '<div class="block"><div class="bl">Model answered</div>'
            f'<div class="bv">{_esc(validation.get("response"))}</div></div>'
        )
    else:
        parts.append(
            '<div class="block"><div class="bl">Generation failed '
            "(an infrastructure fault, not a quality finding)</div>"
            f'<div class="bv">{_esc(validation.get("failure_type"))}: '
            f'{_esc(validation.get("reason"))}</div></div>'
        )

    behavior = classification.get("behavior_signal") or {}
    if behavior.get("signal"):
        parts.append(
            '<div class="block"><div class="bl">Behaviour signal</div>'
            f'<div class="bv">{_esc(behavior.get("signal"))} — '
            f'{_esc(behavior.get("reason"))}</div></div>'
        )

    preflight = result.get("dataset_checks") or {}
    if preflight:
        parts.append('<h3>Dataset checks</h3><p>These scores evaluate the reference material, '
                     'not the generated answer. Context role: '
                     + _esc(preflight.get("context_role")) + '</p><ul>')
        for key, detail in preflight.get("checks", {}).items():
            parts.append('<li>' + _esc(key) + ': ' + _esc(detail.get("status"))
                         + ' — score ' + _score(detail.get("score")) + ': '
                         + _esc(detail.get("reason")) + '</li>')
        parts.append('</ul>')
    parts.append(_metric_rows(result.get("evaluation"), root_cause.get("metric_level")))

    if result.get("human_review"):
        parts.append(
            '<div class="flag">Routed to a human: '
            + _esc("; ".join(result["human_review"]))
            + "</div>"
        )

    parts.append("</div>")
    return "".join(parts)

def generate_html_report(test_case_results, output_dir):
    """Writes index.html into output_dir and returns its path.

    output_dir is the folder reporting/report_generator.py already created
    for this run, so the HTML page and the Excel workbook sit side by side.
    """

    counts = _collect_counts(test_case_results)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not test_case_results:
        body = '<div class="panel"><p class="empty">No test results were available.</p></div>'
    else:
        blocking = counts["blocking"]

        if blocking:
            verdict = (
                f'<p class="verdict fail">Blocked — {len(blocking)} trusted '
                f'metric result{"s" if len(blocking) != 1 else ""} failed</p>'
                '<p class="verdict-note">Only trusted results can fail the run. '
                "Advisory results below are reported but never gate.</p>"
                "<ul>"
                + "".join(
                    f"<li><strong>{_esc(b['metric'])}</strong> "
                    f"({_esc(b['status'])}, score {_score(b['score'])}) on "
                    f"{_esc(b['input'])}</li>"
                    for b in blocking
                )
                + "</ul>"
            )
        else:
            verdict = (
                '<p class="verdict pass">Not blocked — no trusted metric failed</p>'
                '<p class="verdict-note">Advisory and human-review results may still '
                "need your attention; they are shown per test case below.</p>"
            )

        tiles = "".join(
            f'<div class="tile"><div class="n">{value}</div>'
            f'<div class="l">{_esc(label)}</div></div>'
            for label, value in [
                ("Test cases", counts["cases"]),
                ("Verified", counts["classification"]["VERIFIED"]),
                ("Evaluated", counts["classification"]["EVALUATED"]),
                ("Unverifiable", counts["classification"]["UNVERIFIABLE"]),
                ("Need a human", counts["needs_human"]),
            ]
        )

        gating_tiles = "".join(
            f'<div class="tile"><div class="n">{value}</div>'
            f'<div class="l">{_esc(label)}</div></div>'
            for label, value in [
                ("Trusted (can gate)", counts["gating"]["trusted"]),
                ("Advisory (cannot gate)", counts["gating"]["advisory"]),
                ("Human required", counts["gating"]["requires_human_review"]),
                ("Passed", counts["status"]["passed"]),
                ("Failed", counts["status"]["failed"]),
                ("Errored", counts["status"]["error"]),
            ]
        )

        body = (
            f'<div class="panel">{verdict}</div>'
            f"<h2>Test cases</h2><div class=\"grid\">{tiles}</div>"
            f"<h2>Metric results</h2><div class=\"grid\">{gating_tiles}</div>"
            f"<h2>Every test case</h2>"
            + "".join(_case_section(item) for item in test_case_results)
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation run {_esc(generated_at)}</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="wrap">
  <h1>Evaluation run</h1>
  <p class="sub">{_esc(generated_at)}</p>

  <div class="panel">
    <div class="kv">
      <div><span class="k">Generator</span>
        <span class="v">{_esc(settings.GENERATOR_PROVIDER)}/{_esc(settings.GENERATOR_MODEL)}</span></div>
      <div><span class="k">Judge</span>
        <span class="v">{_esc(settings.JUDGE_PROVIDER)}/{_esc(settings.JUDGE_MODEL)}</span></div>
      <div><span class="k">Different generator and judge models</span>
        <span class="v">{"Yes" if uses_different_models() else "No"}</span></div>
    </div>
  </div>

  {body}

  <footer>
    Generated from results the pipeline already computed. This page never runs a
    metric and never calls a judge model, so producing it cannot change a result.
  </footer>
</div>
</body>
</html>
"""

    path = output_dir / "index.html"
    path.write_text(page, encoding="utf-8")

    return str(path)
