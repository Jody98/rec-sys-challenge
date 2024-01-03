import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.KNN import ItemKNNCFRecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz("../input_files/URM_train_plus_validation.npz")
    URM_test = sps.load_npz("../input_files/URM_test.npz")

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    recommender.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.036424892090848766,
                    tversky_beta=0.9961018325655608)

    results, _ = evaluator_test.evaluateRecommender(recommender)
    print("MAP: {}".format(results.loc[10]["MAP"]))

    recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    recommender.fit(topK=11, shrink=25, similarity='asymmetric', asymmetric_alpha=0.02080568072559298)

    results, _ = evaluator_test.evaluateRecommender(recommender)
    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
