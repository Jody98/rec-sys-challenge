import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.FactorizationMachines import LightFMRecommender
from utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    data_file_path = 'input_files/data_train.csv'
    users_file_path = 'input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    recommender = LightFMRecommender.LightFMCFRecommender(URM_train)
    recommender.fit(loss="bpr", sgd_mode="adagrad", n_components=10,
                    item_alpha=0.0, user_alpha=0.0,
                    learning_rate=0.05, num_threads=4)

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("output_files/LightFMSubmission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    for result in results.items():
        print(result)


if __name__ == '__main__':
    __main__()
