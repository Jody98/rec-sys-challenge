import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender, P3alphaRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    RP3beta_Wsparse = RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086,
                                          min_rating=0.2588031389774553,
                                          implicit=True,
                                          normalize_similarity=True)

    RP3_recommender_second = RP3betaRecommender.RP3betaRecommender(URM_train)
    RP3beta_Wsparse_second = RP3_recommender_second.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086,
                                                        min_rating=0.2588031389774553,
                                                        implicit=True,
                                                        normalize_similarity=True)

    P3_recommender = P3alphaRecommender.P3alphaRecommender(URM_train)
    P3alpha_Wsparse = P3_recommender.fit(topK=64, alpha=0.35496275558011753, min_rating=0.1, implicit=True,
                                         normalize_similarity=True)

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_Wsparse = item_recommender.fit(topK=10, shrink=19, similarity='jaccard', normalize=False,
                                        feature_weighting="TF-IDF")

    recommender_object = ItemKNNSimilarityHybridRecommender(URM_train, RP3beta_Wsparse, P3alpha_Wsparse)
    hybrid_Wsparse = recommender_object.fit(topK=56, alpha=0.8126786752572159)

    recommender_object = ItemKNNSimilarityHybridRecommender(URM_train, item_Wsparse, hybrid_Wsparse)
    hybrid_Wsparse = recommender_object.fit(topK=87, alpha=0.3214615056709495)

    recommender_object = ItemKNNSimilarityHybridRecommender(URM_train, RP3beta_Wsparse_second, hybrid_Wsparse)
    hybrid_Wsparse = recommender_object.fit(topK=37, alpha=0.4214615056709495)

    recommended_items = recommender_object.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/HybridSubmission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender_object)

    for result in results.items():
        print(result)


if __name__ == '__main__':
    __main__()
