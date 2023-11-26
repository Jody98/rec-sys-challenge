import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender
from Recommenders.GraphBased.P3alphaRecommender import P3alphaRecommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNCustomSimilarityRecommender import ItemKNNCustomSimilarityRecommender
from Recommenders.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender
from Recommenders.MatrixFactorization.NMFRecommender import NMFRecommender
from Recommenders.MatrixFactorization.PureSVDRecommender import PureSVDItemRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python
from challenge.utils.functions import read_data

data_file_path = '../input_files/data_train.csv'
users_file_path = '../input_files/data_target_users_test.csv'
URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

URM_all = sps.coo_matrix(
    (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
URM_all = URM_all.tocsr()

URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

cutoff_list = [5, 10, 15]

evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

profile_length = np.ediff1d(sps.csr_matrix(URM_train).indptr)
print(profile_length, profile_length.shape)

block_size = int(len(profile_length) * 0.05)

sorted_users = np.argsort(profile_length)

for group_id in range(0, 20):
    start_pos = group_id * block_size
    end_pos = min((group_id + 1) * block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = profile_length[users_in_group]

    print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
        group_id,
        users_in_group.shape[0],
        users_in_group_p_len.mean(),
        np.median(users_in_group_p_len),
        users_in_group_p_len.min(),
        users_in_group_p_len.max()))

MAP_recommender_per_group = {}

collaborative_recommender_class = {"TopPop": TopPop,
                                   "P3alpha": P3alphaRecommender,
                                   "RP3beta": RP3betaRecommender,
                                   "EASE_R": EASE_R_Recommender
                                   }

recommender_object_dict = {}

hyperparameters = {"TopPop": {},
                   "UserKNNCF": {"topK": 100, "shrink": 8, "similarity": "jaccard", "normalize": True,
                                 "feature_weighting": "TF-IDF"},
                   "ItemKNNCF": {"topK": 10, "shrink": 19, "similarity": "jaccard", "normalize": True,
                                 "feature_weighting": "TF-IDF"},
                   "P3alpha": {"topK": 64, "alpha": 0.35496275558011753, "min_rating": 0.1, "implicit": True,
                               "normalize_similarity": True},
                   "RP3beta": {"topK": 30, "alpha": 0.26362900188025656, "beta": 0.17133265585189086,
                               "min_rating": 0.2588031389774553, "implicit": True, "normalize_similarity": True},
                   "PureSVDItem": {"num_factors": 96, "topK": 66},
                   "NMF": {"num_factors": 121, "l1_ratio": 0.6548816817939879, "solver": "multiplicative_update",
                           "init_type": "nndsvda", "beta_loss": "frobenius"},
                   "iALS": {"num_factors": 10, "confidence_scaling": "linear", "alpha": 1.0, "epsilon": 1.0,
                            "reg": 0.001, "init_mean": 0.0, "init_std": 0.1},
                   "SLIMBPR": {"topK": 26, "epochs": 15, "lambda_i": 0.009991555707793169,
                               "lambda_j": 0.004832924438269361, "learning_rate": 0.04032300739781685},
                   "EASE_R": {"topK": 10, "l2_norm": 101, "normalize_matrix": False}
                   }

for label, recommender_class in collaborative_recommender_class.items():
    recommender_object = recommender_class(URM_train)
    recommender_object.fit(**hyperparameters[label])
    recommender_object_dict[label] = recommender_object

cutoff = 10

for group_id in range(0, 20):

    start_pos = group_id * block_size
    end_pos = min((group_id + 1) * block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]

    users_in_group_p_len = profile_length[users_in_group]

    print("Group {}, #users in group {}, average p.len {:.2f}, median {}, min {}, max {}".format(
        group_id,
        users_in_group.shape[0],
        users_in_group_p_len.mean(),
        np.median(users_in_group_p_len),
        users_in_group_p_len.min(),
        users_in_group_p_len.max()))

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

    for label, recommender in recommender_object_dict.items():
        result_df, _ = evaluator_test.evaluateRecommender(recommender)
        if label in MAP_recommender_per_group:
            MAP_recommender_per_group[label].append(result_df.loc[cutoff]["MAP"])
        else:
            MAP_recommender_per_group[label] = [result_df.loc[cutoff]["MAP"]]

_ = plt.figure(figsize=(16, 9))
for label, recommender in recommender_object_dict.items():
    results = MAP_recommender_per_group[label]
    plt.scatter(x=np.arange(0, len(results)), y=results, label=label)
plt.ylabel('MAP')
plt.xlabel('User Group')
plt.legend()
plt.show()
