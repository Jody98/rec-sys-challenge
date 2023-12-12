import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from challenge.utils.functions import read_data
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.SLIM import SLIM_BPR_Python
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    recommender_BPR = SLIM_BPR_Python.SLIM_BPR_Python(URM_train)
    recommender_BPR.fit(topK=10, epochs=15, lambda_i=0.25210476921792674, lambda_j=0.0001,
                        learning_rate=0.05149809958563418)

    results, _ = evaluator.evaluateRecommender(recommender_BPR)
    print("MAP BPR: {}".format(results.loc[10]["MAP"]))

    recommender_EN = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    recommender_EN.fit(topK=216, l1_ratio=0.0032465600313226354, alpha=0.002589066655986645, positive_only=True)

    results, _ = evaluator.evaluateRecommender(recommender_EN)
    print("MAP EN: {}".format(results.loc[10]["MAP"]))

    recommender = DifferentLossScoresHybridRecommender(URM_train, recommender_BPR, recommender_EN)
    recommender.fit(norm=1, alpha=0.2443686520252853)

    results, _ = evaluator.evaluateRecommender(recommender)
    print("MAP Hybrid: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
