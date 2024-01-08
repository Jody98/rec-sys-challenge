import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.SLIM import SLIM_BPR_Python
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/new_URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/new_URM_test.npz')

    recommender = SLIM_BPR_Python.SLIM_BPR_Python(URM_train)
    recommender.fit(topK=10, epochs=15, lambda_i=0.25210476921792674, lambda_j=0.0001,
                    learning_rate=0.05149809958563418)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    best_parameters = {'topK': 9, 'epochs': 114, 'lambda_i': 0.0025338350012924717, 'lambda_j': 1.5050017019467605e-05, 'learning_rate': 0.009006704930203509}

    recommender = SLIM_BPR_Python.SLIM_BPR_Python(URM_train)
    recommender.fit(**best_parameters)

    results, _ = evaluator.evaluateRecommender(recommender)
    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
