import numpy as np
import pandas as pd
import scipy.sparse as sps
from tqdm import tqdm
from xgboost import XGBRanker, plot_importance

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Recommenders.Hybrid.HybridDifferentLoss import DifferentLossScoresHybridRecommender
from Recommenders.GraphBased import P3alphaRecommender, RP3betaRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from challenge.utils.functions import read_data


def __main__():
    cutoff_real = 10
    cutoff_xgb = 20
    submission_file_path = '../output_files/XGBoostSubmission.csv'
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'

    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    n_users, n_items = URM_train.shape

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_all)
    SLIM_recommender.fit(l1_ratio=0.005997129498003861, alpha=0.004503120402472539,
                         positive_only=True, topK=45)

    rp3beta = RP3betaRecommender.RP3betaRecommender(URM_all)
    rp3beta.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                implicit=True, normalize_similarity=True)

    SLIMRP3 = DifferentLossScoresHybridRecommender(URM_all, rp3beta, SLIM_recommender)
    SLIMRP3.fit(norm=1, alpha=0.4969561446020178)

    training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    training_dataframe.index.name = 'UserID'

    for user_id in tqdm(range(n_users)):
        recommendations = SLIMRP3.recommend(user_id, cutoff=cutoff_xgb)
        training_dataframe.loc[user_id, "ItemID"] = recommendations

    training_dataframe = training_dataframe.explode("ItemID")

    URM_validation_coo = sps.coo_matrix(URM_validation)

    correct_recommendations = pd.DataFrame({"UserID": URM_validation_coo.row,
                                            "ItemID": URM_validation_coo.col})

    training_dataframe = pd.merge(training_dataframe, correct_recommendations, on=['UserID', 'ItemID'], how='left',
                                  indicator='Exist')

    training_dataframe["Label"] = training_dataframe["Exist"] == "both"
    training_dataframe.drop(columns=['Exist'], inplace=True)

    topPop = TopPop(URM_all)
    topPop.fit()

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_all)
    item_recommender.fit(topK=10, shrink=19, similarity='jaccard', normalize=False,
                         feature_weighting="TF-IDF")

    EASE_R_recommender = EASE_R_Recommender.EASE_R_Recommender(URM_all)
    EASE_R_recommender.fit(topK=10, l2_norm=101, normalize_matrix=False)

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_all)
    p3alpha.fit(topK=64, alpha=0.35496275558011753, min_rating=0.1, implicit=True,
                normalize_similarity=True)

    other_algorithms = {
        "TopPop": topPop,
        "ItemKNNCF": item_recommender,
        "EASE_R": EASE_R_recommender,
        "P3alpha": p3alpha,
        "RP3beta": rp3beta,
        "SLIM": SLIM_recommender
    }

    training_dataframe = training_dataframe.set_index('UserID')

    for user_id in tqdm(range(n_users)):
        for rec_label, rec_instance in other_algorithms.items():
            item_list = training_dataframe.loc[user_id, "ItemID"].values.tolist()
            all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute=item_list)
            training_dataframe.loc[user_id, rec_label] = all_item_scores[0, item_list]

    training_dataframe = training_dataframe.reset_index()
    training_dataframe = training_dataframe.rename(columns={"index": "UserID"})

    item_popularity = np.ediff1d(sps.csc_matrix(URM_all).indptr)
    training_dataframe['item_popularity'] = item_popularity[training_dataframe["ItemID"].values.astype(int)]

    user_popularity = np.ediff1d(sps.csr_matrix(URM_all).indptr)
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

    y_train = training_dataframe["Label"]
    X_train = training_dataframe.drop(columns=["Label"])
    X_train["UserID"] = X_train["UserID"].astype("category")
    X_train["ItemID"] = X_train["ItemID"].astype("category")

    XGB_model.fit(X_train,
                  y_train,
                  group=groups,
                  verbose=True)

    reranked_df = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    reranked_df.index.name = 'UserID'

    for user_id in tqdm(range(n_users)):
        X_to_predict = X_train[X_train["UserID"] == user_id]
        X_prediction = XGB_model.predict(X_to_predict)
        dict_prediction = dict(zip(X_to_predict["ItemID"], X_prediction))
        dict_prediction = {k: v for k, v in sorted(dict_prediction.items(), key=lambda item: item[1], reverse=True)}
        list_prediction = list(dict_prediction.keys())[:cutoff_real]
        reranked_df.loc[user_id, "ItemID"] = list_prediction

    reranked_df['UserID'] = reranked_df.index

    with open(submission_file_path, 'w') as file:
        file.write('user_id,item_list\n')
        for user_id in tqdm(users_list):
            item_list = reranked_df.loc[user_id, "ItemID"]
            user_string = f"{user_id},{' '.join(map(str, item_list))}\n"
            file.write(user_string)

    plot1 = plot_importance(XGB_model, importance_type='gain', title='Gain')
    plot2 = plot_importance(XGB_model, importance_type='cover', title='Cover')
    plot3 = plot_importance(XGB_model, importance_type='weight', title='Weight (Frequence)')

    plot1.figure.savefig('gain.png')
    plot2.figure.savefig('cover.png')
    plot3.figure.savefig('weight.png')


if __name__ == '__main__':
    __main__()
