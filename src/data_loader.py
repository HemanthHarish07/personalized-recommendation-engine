import pandas as pd
from pathlib import Path


def load_movielens_ratings(data_path: str) -> pd.DataFrame:
    """
    Load MovieLens ratings data and perform basic validation.
    """
    file_path = Path(data_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {file_path}")

    # Load ratings data
    ratings = pd.read_csv(file_path)

    # Required columns for user-item interactions
    required_columns = {"userId", "movieId", "rating"}
    if not required_columns.issubset(ratings.columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")

    # Remove rows with missing values in critical columns
    ratings = ratings.dropna(subset=required_columns)

    return ratings
