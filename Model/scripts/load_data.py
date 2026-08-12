import pandas as pd


def load_data(dataset_path):
    """Load the development dataset from a local path or a HuggingFace parquet URL."""
    print(f"Loading dataset from {dataset_path} ...")
    dev = pd.read_parquet(dataset_path)

    if "contamination_identifier" in dev.columns:
        dev = dev.drop(columns=["contamination_identifier"])

    return dev
