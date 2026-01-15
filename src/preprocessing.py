import pandas as pd


def filter_positive_interactions(
    ratings: pd.DataFrame, min_rating: float = 4.0
) -> pd.DataFrame:
    """
    Filter ratings to keep only positive user-item interactions.
    """
    if "rating" not in ratings.columns:
        raise ValueError("Input DataFrame must contain a 'rating' column")

    # Keep only strong positive feedback
    positive_interactions = ratings[ratings["rating"] >= min_rating].copy()

    return positive_interactions
