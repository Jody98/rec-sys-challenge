import numpy as np
import scipy.sparse as sps
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def create_new_users(URM_train, users_list, num_users_to_create=100, similarity_threshold=0.8):
    user_similarity = cosine_similarity(URM_train, dense_output=False)
    URM_train_augmented = URM_train.copy()

    for _ in tqdm(range(num_users_to_create)):
        random_user_index = np.random.choice(len(users_list))
        model_user = users_list[random_user_index]

        similarities = user_similarity[model_user, :].toarray().ravel()

        similar_users = np.where(similarities > similarity_threshold)[0]

        if similar_users.size > 0:
            random_similar_user = np.random.choice(similar_users)
            URM_train_augmented[random_similar_user, :] += URM_train[model_user, :]

    return URM_train_augmented


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    num_interactions = URM_train.nnz
    num_interactions_test = URM_test.nnz

    URM_train_augmented = create_new_users(URM_train, users_list, num_users_to_create=1000, similarity_threshold=0.18)
    num_interactions_augmented = URM_train_augmented.nnz

    topK = [30]

    for topk in topK:
        recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
        recommender.fit(topK=topk, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)

        recommended_items = recommender.recommend(users_list, cutoff=10)
        recommendations = []
        for i in zip(users_list, recommended_items):
            recommendation = {"user_id": i[0], "item_list": i[1]}
            recommendations.append(recommendation)

        generate_submission_csv("../output_files/RP3betaSubmission.csv", recommendations)

        evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
        results, _ = evaluator.evaluateRecommender(recommender)

        print("TopK: {}".format(topk))
        print("MAP: {}".format(results.loc[10]["MAP"]))

        recommender = RP3betaRecommender.RP3betaRecommender(URM_train_augmented)
        recommender.fit(topK=topk, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)

        recommended_items = recommender.recommend(users_list, cutoff=10)
        recommendations = []
        for i in zip(users_list, recommended_items):
            recommendation = {"user_id": i[0], "item_list": i[1]}
            recommendations.append(recommendation)

        generate_submission_csv("../output_files/RP3betaSubmission.csv", recommendations)

        evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
        results, _ = evaluator.evaluateRecommender(recommender)

        print("TopK: {}".format(topk))
        print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
