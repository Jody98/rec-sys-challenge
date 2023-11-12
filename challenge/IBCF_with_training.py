import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from recommenders.collaborative_filtering_recommender import ItemKNNCFRecommender
from utils.functions import read_data, evaluate_algorithm, generate_submission_csv


def __main__():
    data_file_path = 'input_files/data_train.csv'
    users_file_path = 'input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    topk = 10
    shrink = 10
    similarity = 'jaccard'

    recommender = ItemKNNCFRecommender(URM_train)
    recommender.fit(shrink=shrink, topK=topk, similarity=similarity)

    recommendations = []

    for user_id in users_list:
        recommendation = recommender.recommend(user_id, at=10)[0]
        recommendations.append(recommendation)

    generate_submission_csv("output_files/IBCF_submission.csv", recommendations)
    print("similarity: {}".format(s))
    evaluate_algorithm(URM_test, recommender, at=10)


if __name__ == '__main__':
    __main__()
