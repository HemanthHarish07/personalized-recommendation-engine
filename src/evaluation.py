import random


def train_test_split_by_user(interactions, test_size=1, seed=42):
    """
    Split interactions by holding out items per user.
    """
    random.seed(seed)
    train_data = []
    test_data = []

    for user_id, user_interactions in interactions.groupby("userId"):
        items = list(user_interactions["movieId"])
        if len(items) <= test_size:
            continue

        test_items = random.sample(items, test_size)

        for _, row in user_interactions.iterrows():
            if row["movieId"] in test_items:
                test_data.append(row)
            else:
                train_data.append(row)

    return train_data, test_data


def hit_rate_at_k(model, train_interactions, test_interactions, k=10):
    """
    Compute Hit Rate@K for a recommender model.
    """
    hits = 0
    users = 0

    model.fit()

    # Group test interactions by user
    test_by_user = (
        test_interactions.groupby("userId")["movieId"].apply(list).to_dict()
    )

    for user_id, true_items in test_by_user.items():
        try:
            recommendations = model.recommend(user_id, top_n=k)
        except TypeError:
            # Popularity model does not require user_id
            recommendations = model.recommend(top_n=k)

        if any(item in recommendations for item in true_items):
            hits += 1
        users += 1

    return hits / users if users > 0 else 0.0

