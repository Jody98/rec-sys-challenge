import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from challenge.utils.functions import read_data, generate_submission_csv


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz("../input_files/URM_train_plus_validation.npz")
    URM_test = sps.load_npz("../input_files/URM_test.npz")

    weights = [0.0075]

    for weight in weights:
        recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
        recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True, tail=True, tail_weight=weight)

        recommended_items = recommender.recommend(users_list, cutoff=10)
        recommendations = []
        for i in zip(users_list, recommended_items):
            recommendation = {"user_id": i[0], "item_list": i[1]}
            recommendations.append(recommendation)

        generate_submission_csv("../output_files/SLIMElasticNetSubmission.csv", recommendations)

        evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
        results, _ = evaluator.evaluateRecommender(recommender)

        print("Weight: {}".format(weight))
        print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
