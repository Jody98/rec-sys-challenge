import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train_augmented.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')

    recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                    implicit=True, normalize_similarity=True)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    recommender.fit(topK=29, alpha=0.177392841911343, beta=0.14355849072065044, min_rating=0.005923820399700632,
                    implicit=True, normalize_similarity=True, tail=True, tail_weight=0.02903606290641228)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
