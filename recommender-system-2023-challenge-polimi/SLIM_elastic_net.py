import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.KNN import ItemKNNCFRecommender, UserKNNCFRecommender
from Recommenders.SLIM import SLIMElasticNetRecommender
from utils.functions import read_data, generate_submission_csv
from Recommenders.GraphBased import RP3betaRecommender


def __main__():
    URM_all_dataframe, users_list = read_data()

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    recommender.fit()

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/submission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)

    recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    recommender.fit()

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

    recommender = UserKNNCFRecommender.UserKNNCFRecommender(URM_train)
    recommender.fit()

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/BaseMatrixFactorizationRecommender.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)

    recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    recommender.fit()

    recommended_items = recommender.recommend(users_list, cutoff=10)

    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/RP3betaRecommender.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)


if __name__ == '__main__':
    __main__()
