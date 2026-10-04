from utils.report_generator import (
    generate_excel_report,
    results_data as legacy_results_data
)
from reporting.report_generator import (
    generate_report,
    results_data as pipeline_results_data
)

def pytest_sessionfinish(session, exitstatus):

    print("\nGenerating Excel Report...")
    generate_excel_report(legacy_results_data)

    print("\nGenerating Evaluation Report...")
    generate_report(pipeline_results_data)