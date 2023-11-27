import scipy.sparse as sps
from challenge.experiments.cython_SLIM_MSE import train_multiple_epochs

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from Recommenders.KNN import ItemKNNCustomSimilarityRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def create_W_sparse(URM_train):
    learning_rate = 1e-4
    epochs = 60
    regularization = 0.025

    item_item_S, loss, samples_per_second = train_multiple_epochs(URM_train=URM_train,
                                                                  learning_rate_input=learning_rate,
                                                                  regularization_2_input=regularization,
                                                                  n_epochs=epochs)

    W_sparse = sps.csr_matrix(item_item_S)

    return W_sparse


def __main__():
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    SLIMMSE_Wsparse = create_W_sparse(URM_all)

    recommender = RP3betaRecommender.RP3betaRecommender(URM_all)
    RP3_Wsparse = recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086,
                                  min_rating=0.2588031389774553,
                                  implicit=True,
                                  normalize_similarity=True)

    alpha = 0.1
    Wsparse = alpha * SLIMMSE_Wsparse + (1 - alpha) * RP3_Wsparse

    recommender = ItemKNNCustomSimilarityRecommender.ItemKNNCustomSimilarityRecommender(URM_all)
    recommender.fit(W_sparse=Wsparse, selectTopK=True, topK=10)

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/HybridSLIMMSESubmission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)


if __name__ == '__main__':
    __main__()
