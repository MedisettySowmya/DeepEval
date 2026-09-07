import json
import pandas as pd


def load_qa_boat_datasets(path):

    excel_file = pd.ExcelFile(path)
    dataset = []

    for sheet_name in excel_file.sheet_names:

        df = pd.read_excel(path, sheet_name=sheet_name)

        for _, row in df.iterrows():
            
            raw_ground_truth = row.get("Ground_Truth")
            ground_truth = None

            if pd.notna(raw_ground_truth) and str(raw_ground_truth).strip():
                ground_truth = json.loads(raw_ground_truth)

            dataset.append({
                "id": row["ID"],
                "sheet_name": sheet_name,
                "requirement": row["Requirement"],
                "acceptance_criteria": row["Acceptance Criteria"],
                "ground_truth": ground_truth
            })

    return dataset