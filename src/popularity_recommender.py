import pandas as pd


class PopularityRecommender:
    """
    Recommend items based on overall popularity.
    """

    def __init__(self, interactions: pd.DataFrame):
        if not {"userId", "movieId"}.issubset(interactions.columns):
            raise ValueError("Interactions must contain userId and movieId columns")

        self.interactions = interactions
        self.popular_items = None

    def fit(self):
        """
        Compute item popularity based on interaction count.
        """
        self.popular_items = (
            self.interactions.groupby("movieId")["userId"]
            .count()
            .sort_values(ascending=False)
        )

    def recommend(self, top_n: int = 10):
        """
        Return top-N most popular items.
        """
        if self.popular_items is None:
            raise RuntimeError("Model must be fitted before calling recommend")

        return self.popular_items.head(top_n).index.tolist()
