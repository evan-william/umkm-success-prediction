from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "Age",
    "Education",
    "Initial_Capital",
    "Financial_Record_Keeping",
    "Internet_Usage",
    "Business_Plan",
    "Marketing_Effort",
    "Partnership",
    "Parent_Business_Experience",
    "Industry_Experience",
    "Owner_Gender",
    "Professional_Advice",
    "Success",
]


def test_dataset_contract():
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "umkm_success.csv"
    assert dataset_path.exists(), "Official competition dataset is missing."

    df = pd.read_csv(dataset_path)

    assert list(df.columns) == EXPECTED_COLUMNS
    assert df.shape == (250, 13)
    assert df.isna().sum().sum() == 0
    assert set(df["Success"].unique()) == {0, 1}
