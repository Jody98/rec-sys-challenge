import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import P3alphaRecommender
from challenge.utils.functions import read_data, generate_submission_csv, evaluate_algorithm


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz("../input_files/URM_train_plus_validation.npz")
    URM_test = sps.load_npz("../input_files/URM_test.npz")

    topK = [40]
    min_ratings = [0.01]

    for topk in topK:
        for min_rating in min_ratings:

            recommender = P3alphaRecommender.P3alphaRecommender(URM_train)
            recommender.fit(topK=topk, alpha=0.3119217553589628, min_rating=min_rating, implicit=True,
                            normalize_similarity=True)

            recommended_items = recommender.recommend(users_list, cutoff=10)
            recommendations = []
            for i in zip(users_list, recommended_items):
                recommendation = {"user_id": i[0], "item_list": i[1]}
                recommendations.append(recommendation)

            generate_submission_csv("../output_files/P3alphaSubmission.csv", recommendations)

            evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
            results, _ = evaluator.evaluateRecommender(recommender)

            evaluate_algorithm(URM_test, recommender, cutoff_list[0])

            print("TopK: {}".format(topk))
            print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
