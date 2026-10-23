from reporting.report_generator import (
    generate_report,
    results_data as pipeline_results_data
)

def pytest_sessionfinish(session, exitstatus):

    if not pipeline_results_data:
        print("\nNo test results collected -- skipping report generation.")
        return

    print("\nGenerating Evaluation Report...")
    generate_report(pipeline_results_data)
