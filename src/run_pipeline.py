from data_loader import load_movielens_ratings
from preprocessing import filter_positive_interactions
from popularity_recommender import PopularityRecommender
from user_based_cf import UserBasedCF
from item_based_cf import ItemBasedCF
from evaluation import train_test_split_by_user, hit_rate_at_k

import pandas as pd


def main():
    ratings = load_movielens_ratings("data/ml-100k/u.data")

    interactions = filter_positive_interactions(ratings)

    train_data, test_data = train_test_split_by_user(interactions)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    print("Evaluating Popularity Recommender...")
    pop_model = PopularityRecommender(train_df)
    print("Hit Rate@10:", hit_rate_at_k(pop_model, train_df, test_df))

    print("Evaluating User-Based CF...")
    user_cf = UserBasedCF(train_df)
    print("Hit Rate@10:", hit_rate_at_k(user_cf, train_df, test_df))

    print("Evaluating Item-Based CF...")
    item_cf = ItemBasedCF(train_df)
    print("Hit Rate@10:", hit_rate_at_k(item_cf, train_df, test_df))


if __name__ == "__main__":
    main()
