import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm
from xgboost import XGBRanker, plot_importance

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import P3alphaRecommender, RP3betaRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_real = 10
    cutoff_xgb = 20
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train_validation, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train_validation, train_percentage=0.80)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=[cutoff_real])

    n_users, n_items = URM_train.shape

    slim = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    slim.fit(l1_ratio=0.005997129498003861, alpha=0.004503120402472539,
             positive_only=True, topK=45)

    training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    training_dataframe.index.name = 'UserID'

    for user_id in tqdm(range(n_users)):
        recommendations = slim.recommend(user_id, cutoff=cutoff_xgb)
        training_dataframe.loc[user_id, "ItemID"] = recommendations

    training_dataframe = training_dataframe.explode("ItemID")

    URM_validation_coo = sps.coo_matrix(URM_validation)

    correct_recommendations = pd.DataFrame({"UserID": URM_validation_coo.row,
                                            "ItemID": URM_validation_coo.col})

    training_dataframe = pd.merge(training_dataframe, correct_recommendations, on=['UserID', 'ItemID'], how='left',
                                  indicator='Exist')

    training_dataframe["Label"] = training_dataframe["Exist"] == "both"
    training_dataframe.drop(columns=['Exist'], inplace=True)

    topPop = TopPop(URM_train)
    topPop.fit()

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    p3alpha.fit(topK=64, alpha=0.35496275558011753, min_rating=0.1, implicit=True,
                normalize_similarity=True)

    rp3beta = RP3betaRecommender.RP3betaRecommender(URM_train)
    rp3beta.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                implicit=True, normalize_similarity=True)

    other_algorithms = {
        "TopPop": topPop,
        "P3alpha": p3alpha,
        "RP3beta": rp3beta,
    }

    training_dataframe = training_dataframe.set_index('UserID')

    for user_id in tqdm(range(n_users)):
        for rec_label, rec_instance in other_algorithms.items():
            item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()
            all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute=item_list)
            training_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list]

    training_dataframe = training_dataframe.reset_index()
    training_dataframe = training_dataframe.rename(columns={"index": "UserID"})

    item_popularity = np.ediff1d(sps.csc_matrix(URM_train).indptr)
    training_dataframe['item_popularity'] = item_popularity[training_dataframe["ItemID"].values.astype(int)]

    user_popularity = np.ediff1d(sps.csr_matrix(URM_train).indptr)
    training_dataframe['user_profile_len'] = user_popularity[training_dataframe["UserID"].values.astype(int)]

    groups = training_dataframe.groupby("UserID").size().values

    n_estimators = 50
    learning_rate = 1e-1
    reg_alpha = 1e-1
    reg_lambda = 1e-1
    max_depth = 5
    max_leaves = 0
    grow_policy = "depthwise"
    objective = "pairwise"
    booster = "gbtree"
    use_user_profile = False
    random_seed = None

    XGB_model = XGBRanker(objective='rank:{}'.format(objective),
                          n_estimators=int(n_estimators),
                          random_state=random_seed,
                          learning_rate=learning_rate,
                          reg_alpha=reg_alpha,
                          reg_lambda=reg_lambda,
                          max_depth=int(max_depth),
                          max_leaves=int(max_leaves),
                          grow_policy=grow_policy,
                          verbosity=0,
                          enable_categorical=True,
                          booster=booster,
                          )

    training_dataframe.drop(columns=["Label"])

    y_train = training_dataframe["Label"]
    X_train = training_dataframe.drop(columns=["Label"])
    X_train["UserID"] = X_train["UserID"].astype("category")
    X_train["ItemID"] = X_train["ItemID"].astype("category")

    XGB_model.fit(X_train,
                  y_train,
                  group=groups,
                  verbose=True)

    # Let's say I want to compute the prediction for a group of user-item pairs, for simplicity I will use a slice of the data used
    # for training because it already contains all the features
    X_to_predict = X_train[X_train["UserID"] == 10]

    print(XGB_model.predict(X_to_predict))

    plot1 = plot_importance(XGB_model, importance_type='gain', title='Gain')
    plot2 = plot_importance(XGB_model, importance_type='cover', title='Cover')
    plot3 = plot_importance(XGB_model, importance_type='weight', title='Weight (Frequence)')

    plot1.figure.savefig('gain.png')
    plot2.figure.savefig('cover.png')
    plot3.figure.savefig('weight.png')


if __name__ == '__main__':
    __main__()
