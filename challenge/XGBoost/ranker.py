import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBRanker
import xgboost as xgb

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Recommenders.GraphBased import P3alphaRecommender, RP3betaRecommender
from Recommenders.KNN import ItemKNNCFRecommender
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

    rp3beta = RP3betaRecommender.RP3betaRecommender(URM_train)
    rp3beta.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                implicit=True, normalize_similarity=True)

    training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    training_dataframe.index.name = 'UserID'

    user_recommendations_items = []
    user_recommendations_user_id = []

    for user_id in tqdm(range(n_users)):
        recommendations = rp3beta.recommend(user_id, cutoff=cutoff_xgb)
        training_dataframe.loc[user_id, "ItemID"] = recommendations
        user_recommendations_items.extend(recommendations)
        user_recommendations_user_id.extend([user_id] * len(recommendations))

    target = []
    count = 0

    for user_id, item_id in zip(user_recommendations_user_id, user_recommendations_items):
        target.append(1 if count % cutoff_xgb < 10 else 0)
        count = count + 1

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

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=10, shrink=19, similarity='jaccard', normalize=False,
                         feature_weighting="TF-IDF")

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    p3alpha.fit(topK=64, alpha=0.35496275558011753, min_rating=0.1, implicit=True,
                normalize_similarity=True)

    other_algorithms = {
        "TopPop": topPop,
        "ItemKNNCF": item_recommender,
        "P3alpha": p3alpha,
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

    n_estimators = 100
    learning_rate = 1e-1
    reg_alpha = 1e-1
    reg_lambda = 1e-1
    max_depth = 5
    max_leaves = 0
    grow_policy = "depthwise"
    objective = "pairwise"
    booster = "gbtree"
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

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    groups_train = groups[:10420]

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    params = {
        'max_depth': 3,
        'eta': 0.3,
        'objective': 'multi:softprob',
        'num_class': 2,
        'eval_metric': 'merror'
    }

    num_round = 1000

    model = xgb.train(params,
                      dtrain,
                      num_round,
                      verbose_eval=2,
                      evals=[(dtrain, 'train'), (dval, 'validation')],
                      early_stopping_rounds=20)

    '''XGB_model.fit(X_train,
                  y_train,
                  group=groups_train,
                  verbose=True)

    y_pred = XGB_model.predict(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    balanced_threshold_index = np.argmax(precision + recall)
    balanced_threshold = thresholds[balanced_threshold_index] * 1.8
    y_pred_balanced = (y_pred > balanced_threshold).astype(int)

    precision = precision_score(y_test, y_pred_balanced)
    recall = recall_score(y_test, y_pred_balanced)
    average_precision = average_precision_score(y_test, y_pred_balanced)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'Mean Average Precision (MAP): {average_precision}')'''


'''reranked_df = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
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
plot3.figure.savefig('weight.png')'''

if __name__ == '__main__':
    __main__()
