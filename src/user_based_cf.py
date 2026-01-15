import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserBasedCF:
    """
    User-based collaborative filtering recommender.
    """

    def __init__(self, interactions: pd.DataFrame):
        if not {"userId", "movieId"}.issubset(interactions.columns):
            raise ValueError("Interactions must contain userId and movieId columns")

        self.interactions = interactions
        self.user_item_matrix = None
        self.user_similarity = None

    def fit(self):
        """
        Build user-item matrix and compute user similarity.
        """
        self.user_item_matrix = pd.pivot_table(
            self.interactions,
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0,
        )

        self.user_similarity = cosine_similarity(self.user_item_matrix)

    def recommend(self, user_id: int, top_n: int = 10):
        """
        Recommend items for a given user based on similar users.
        """
        if self.user_similarity is None:
            raise RuntimeError("Model must be fitted before calling recommend")

        user_idx = self.user_item_matrix.index.get_loc(user_id)
        similarity_scores = self.user_similarity[user_idx]

        similar_users = similarity_scores.argsort()[::-1][1:]

        user_items = set(
            self.interactions[self.interactions["userId"] == user_id]["movieId"]
        )

        recommendations = []

        for other_user_idx in similar_users:
            other_user_id = self.user_item_matrix.index[other_user_idx]
            other_items = set(
                self.interactions[
                    self.interactions["userId"] == other_user_id
                ]["movieId"]
            )

            for item in other_items - user_items:
                recommendations.append(item)

            if len(recommendations) >= top_n:
                break

        return recommendations[:top_n]
