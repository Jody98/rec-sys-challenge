import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.KNN import ItemKNNCFRecommender
from utils.functions import read_data, generate_submission_csv
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.GraphBased import P3alphaRecommender
from Recommenders.MatrixFactorization import PureSVDRecommender, IALSRecommender, NMFRecommender, SVDFeatureRecommender
from Recommenders.Neural import MultVAERecommender


def __main__():
    URM_all_dataframe, users_list = read_data()

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    recommender.fit(topK=200, shrink=100, similarity='cosine')

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

    '''recommender = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    recommender.fit()

    recommended_items = recommender.recommend(users_list, cutoff=10)

    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/EASE_R_Recommender.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)

    recommender = P3alphaRecommender.P3alphaRecommender(URM_train)
    recommender.fit()

    recommended_items = recommender.recommend(users_list, cutoff=10)

    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/P3alphaRecommender.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)

    recommender = PureSVDRecommender.PureSVDRecommender(URM_train)
    recommender.fit()

    recommended_items = recommender.recommend(users_list, cutoff=10)

    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/PureSVDRecommender.csv", recommendations)

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

    generate_submission_csv("output_files/IALSRecommender.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)

    recommender = NMFRecommender.NMFRecommender(URM_train)
    recommender.fit()

    recommended_items = recommender.recommend(users_list, cutoff=10)

    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/NMFRecommender.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)

    recommender = SVDFeatureRecommender.SVDFeature(URM_train=URM_train)
    recommender.fit()

    recommended_items = recommender.recommend(users_list, cutoff=10)

    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/SVDFeatureRecommender.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)'''

    recommender = MultVAERecommender.MultVAERecommender(URM_train)
    recommender.fit()

    recommended_items = recommender.recommend(users_list, cutoff=10)

    recommendations = []

    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/MultVAERecommender.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[10])
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)



if __name__ == '__main__':
    __main__()
