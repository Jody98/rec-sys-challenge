import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from utils.functions import read_data, generate_submission_csv

import numpy as np
from scipy.sparse import csr_matrix


def augment_urm(urm: csr_matrix, oversample_rate: float = 0.1):
    """
    Augment the URM by oversampling positive interactions.

    Parameters:
    urm (csr_matrix): The User-Item Matrix to augment.
    oversample_rate (float): The rate at which to oversample positive interactions.

    Returns:
    csr_matrix: The augmented User-Item Matrix.
    """
    # Get the indices of positive interactions
    positive_interactions = urm.nonzero()

    # Calculate the number of positive interactions to oversample
    num_oversamples = int(len(positive_interactions[0]) * oversample_rate)

    # Randomly select positive interactions to oversample
    oversample_indices = np.random.choice(len(positive_interactions[0]), size=num_oversamples)

    # Get the user-item pairs to oversample
    oversample_user_items = (positive_interactions[0][oversample_indices], positive_interactions[1][oversample_indices])

    # Create a new URM with the oversampled interactions
    urm_oversampled = urm.copy()
    urm_oversampled[oversample_user_items] += 1

    return urm_oversampled


def __main__():
    cutoff_list = [10]
    data_file_path = 'input_files/data_train.csv'
    users_file_path = 'input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                    implicit=True,
                    normalize_similarity=True)

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/RP3betaSubmission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)


if __name__ == '__main__':
    __main__()
