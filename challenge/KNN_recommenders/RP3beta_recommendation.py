import scipy.sparse as sps
import pandas as pd

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from challenge.utils.functions import read_data, generate_submission_csv, evaluate_algorithm, dataset_splits, \
    preprocess_data, generate_submission_csv, write_submission, prepare_submission


def __main__():
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    filename = "RP3betaRecommender_best_model.zip"
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    dataframe1 = pd.read_csv(users_file_path)
    dataframe2 = pd.read_csv(data_file_path)

    dataframe2, dataframe1 = preprocess_data(dataframe2, dataframe1)

    mapping_to_item_id = dict(zip(dataframe2.mapped_item_id, dataframe2.col))

    URM_train, URM_validation, URM_test, URM_all, URM_train_validation = dataset_splits(dataframe2,
                                                                                        num_users=12859,
                                                                                        num_items=22222,
                                                                                        validation_percentage=0.20,
                                                                                        testing_percentage=0.20)

    recommender = RP3betaRecommender.RP3betaRecommender(URM_train_validation)
    recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                    implicit=True, normalize_similarity=True)
    # recommender.load_model(folder_path, filename)

    submission = prepare_submission(dataframe1, dataframe2, recommender, mapping_to_item_id)
    write_submission(submission)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    evaluate_algorithm(URM_test, recommender, cutoff_list[0])

    print("MAP: {}".format(results.loc[10]["MAP"]))

    URM_train = sps.load_npz("../input_files/URM_train_plus_validation.npz")
    URM_test = sps.load_npz("../input_files/URM_test.npz")

    recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                    implicit=True, normalize_similarity=True)

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/RP3betaSubmission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    evaluate_algorithm(URM_test, recommender, cutoff_list[0])

    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
