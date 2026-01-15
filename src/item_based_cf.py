import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemBasedCF:
    """
    Item-based collaborative filtering recommender.
    """

    def __init__(self, interactions: pd.DataFrame):
        if not {"userId", "movieId"}.issubset(interactions.columns):
            raise ValueError("Interactions must contain userId and movieId columns")

        self.interactions = interactions
        self.item_user_matrix = None
        self.item_similarity = None

    def fit(self):
        """
        Build item-user matrix and compute item similarity.
        """
        self.item_user_matrix = pd.pivot_table(
            self.interactions,
            index="movieId",
            columns="userId",
            values="rating",
            fill_value=0,
        )

        self.item_similarity = cosine_similarity(self.item_user_matrix)

    def recommend(self, user_id: int, top_n: int = 10):
        """
        Recommend items for a given user based on item similarity.
        """
        if self.item_similarity is None:
            raise RuntimeError("Model must be fitted before calling recommend")

        user_items = self.interactions[
            self.interactions["userId"] == user_id
        ]["movieId"].unique()

        scores = {}

        for item in user_items:
            if item not in self.item_user_matrix.index:
                continue

            item_idx = self.item_user_matrix.index.get_loc(item)
            similarity_scores = self.item_similarity[item_idx]

            for idx, score in enumerate(similarity_scores):
                candidate_item = self.item_user_matrix.index[idx]
                if candidate_item in user_items:
                    continue

                scores[candidate_item] = scores.get(candidate_item, 0) + score

        ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [item for item, _ in ranked_items[:top_n]]
