import pandas as pd
from pathlib import Path


def load_movielens_ratings(data_path: str) -> pd.DataFrame:
    """
    Load MovieLens 100K ratings data and perform basic validation.
    """
    file_path = Path(data_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at {file_path}")

    ratings = pd.read_csv(
        file_path,
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
    )

    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])

    return ratings
