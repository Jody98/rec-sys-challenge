import scipy.sparse as sps
import numpy as np

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.Neural import MultVAERecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_test = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    recommender = MultVAERecommender.MultVAERecommender_PyTorch_OptimizerMask(URM_train)
    recommender.fit(
        epochs=2,
        batch_size=924,
        total_anneal_steps=1480,
        learning_rate=0.0059,
        l2_reg=0.015803595128175245,
        dropout=0.002701341132792915,
        anneal_cap=0.5182821933828626,
        sgd_mode="adagrad",
        encoding_size=2961,
        next_layer_size_multiplier=19,
        max_parameters=np.inf,
        max_n_hidden_layers=7
    )

    # MAP: 0.013391331760494597

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)
    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
