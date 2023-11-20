import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.MatrixFactorization import IALSRecommender
from Recommenders.SLIM import SLIM_BPR_Python
from utils.functions import read_data, generate_submission_csv


def __main__():
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    recommender.fit(topK=10, shrink=19, similarity='jaccard', normalize=False)

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/ItemKNNSubmission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)

    recommender = SLIM_BPR_Python.SLIM_BPR_Python(URM_train)
    recommender.fit(topK=10)

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/SLIM_BPR_Python_submission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)

    recommender = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    recommender.fit()

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/EASE_R_Recommender_submission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)

    recommender = IALSRecommender.IALSRecommender(URM_train)
    recommender.fit()

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/IALSRecommender_submission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)


if __name__ == '__main__':
    __main__()
