import scipy.sparse as sps
from matplotlib import pyplot

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from recommenders.collaborative_filtering_recommender import ItemKNNCFRecommender
from utils.functions import read_data, evaluate_algorithm, generate_submission_csv


def __main__():
    URM_all_dataframe, users_list = read_data()

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    evaluator_validation = EvaluatorHoldout(URM_validation, cutoff_list=[10])
    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[10])

    recommender = ItemKNNCFRecommender(URM_train)
    # il miglior fit è dato da topK=10 e shrink=10.0
    recommender.fit(shrink=10.0, topK=10)

    recommendations = []

    for user_id in users_list:
        recommendation = recommender.recommend(user_id, at=10)[0]
        recommendations.append(recommendation)

    generate_submission_csv("output_files/IBCF_submission.csv", recommendations)

    evaluate_algorithm(URM_test, recommender, at=10)


if __name__ == '__main__':
    __main__()
