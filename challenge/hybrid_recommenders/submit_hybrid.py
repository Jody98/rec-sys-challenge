import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Recommenders.GraphBased import RP3betaRecommender
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import generate_submission_csv
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_all)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_all)
    SLIM_recommender.fit(topK=46, l1_ratio=0.005997129498003861, alpha=0.004503120402472538, positive_only=True)
    SLIM_Wsparse = SLIM_recommender.W_sparse

    SLIMRP3DifferentLoss = DifferentLossScoresHybridRecommender(URM_all, RP3_recommender, SLIM_recommender)
    SLIMRP3DifferentLoss.fit(norm=2, alpha=0.4304989217384739)

    recommended_items = SLIMRP3DifferentLoss.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/SLIMRP3Normalized_submission.csv", recommendations)


if __name__ == '__main__':
    __main__()
