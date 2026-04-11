import pandas as pd
import yaml
from loguru import logger


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Raw data shape: {df.shape}")

    df = df.dropna(subset=["company", "review", "issue"])

    df["review"] = df["review"].str.strip()
    df["company"] = df["company"].str.strip()
    df["issue"] = df["issue"].str.strip()

    df = df.drop_duplicates(subset=["review"])

    df = df[df["review"].str.len() >= 20]

    df["company"] = df["company"].str.lower()
    df["issue"] = df["issue"].str.lower()
    df["domain"] = df["domain"].str.lower()

    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def main():
    params = load_params()
    raw_path = params["data"]["raw_path"]

    logger.info(f"Loading data from {raw_path}")
    df = pd.read_csv(raw_path)

    df_clean = clean_data(df)

    output_path = "data/cleaned.csv"
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")


if __name__ == "__main__":
    main()