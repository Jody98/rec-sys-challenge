import numpy as np
import pandas as pd
import scipy.sparse as sps
import xgboost as xgb
from hyperopt import fmin, tpe, Trials, STATUS_OK, hp
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import P3alphaRecommender, RP3betaRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from challenge.utils.functions import read_data


def evaluate_model(model, X_val, y_val, groups_val, cutoff):
    y_pred = model.predict(X_val)

    ap_scores = []
    start = 0
    for group_size in groups_val:
        end = start + group_size
        actual_items = set(y_val[start:end])
        predicted_items = y_pred[start:end]
        predicted_items_sorted = [x for _, x in sorted(zip(predicted_items, range(len(predicted_items))), reverse=True)]
        ap_scores.append(average_precision(cutoff, actual_items, predicted_items_sorted))
        start = end

    return np.mean(ap_scores)


def average_precision(at_k, true_items, predicted_items):
    if len(predicted_items) > at_k:
        predicted_items = predicted_items[:at_k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted_items):
        if p in true_items and p not in predicted_items[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not true_items:
        return 0.0

    return score / min(len(true_items), at_k)


def __main__():
    cutoff_real = 10
    cutoff_xgb = 20
    cutoff_list = [cutoff_real]
    submission_file_path = '../output_files/XGBoostSubmission.csv'
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'

    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    evaluator = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)

    space = {
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'reg_alpha': hp.uniform('reg_alpha', 0, 5),
        'reg_lambda': hp.uniform('reg_lambda', 0, 5),
        'max_depth': hp.choice('max_depth', [5, 10, 15]),
        'max_leaves': hp.choice('max_leaves', [0, 5, 10]),
        'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),
        'booster': 'gbtree',
        'enable_categorical': True
    }

    n_users, n_items = URM_train.shape

    rp3beta = RP3betaRecommender.RP3betaRecommender(URM_train)
    rp3beta.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                implicit=True, normalize_similarity=True)
    RP3_Wsparse = rp3beta.W_sparse

    results, _ = evaluator.evaluateRecommender(rp3beta)
    print("RP3betaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))
    baseline_map = results.loc[10]["MAP"]

    training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    training_dataframe.index.name = 'UserID'

    for user_id in tqdm(range(n_users)):
        recommendations = rp3beta.recommend(user_id, cutoff=cutoff_xgb)
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

    y_train = training_dataframe["Label"]
    X_train = training_dataframe.drop(columns=["Label"])
    X_train["UserID"] = X_train["UserID"].astype("category")
    X_train["ItemID"] = X_train["ItemID"].astype("category")

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    groups_val = X_val.groupby("UserID").size().values
    groups = X_train.groupby("UserID").size().values

    def obj(params):
        model = xgb.XGBRanker(objective='rank:pairwise', **params)
        model.fit(X_train, y_train, group=groups, verbose=True)

        y_pred = model.predict(X_val)

        ap_scores = []
        start = 0
        for group_size in groups_val:  # groups_val è l'array che indica la dimensione di ciascun gruppo nel set di validazione
            end = start + group_size
            actual_items = set(y_val[start:end])
            predicted_items = y_pred[start:end]
            predicted_items_sorted = [x for _, x in
                                      sorted(zip(predicted_items, range(len(predicted_items))), reverse=True)]
            ap_scores.append(average_precision(cutoff_real, actual_items, predicted_items_sorted))
            start = end

        score = np.mean(ap_scores)

        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_params = fmin(fn=obj,
                       space=space,
                       algo=tpe.suggest,
                       max_evals=2,
                       trials=trials)

    print("Migliori parametri: ", best_params)

    model_optimized = xgb.XGBRanker(objective='rank:pairwise', **best_params, enable_categorical=True)
    model_optimized.fit(X_train, y_train, group=groups, verbose=True)
    optimized_map = evaluate_model(model_optimized, X_val, y_val, groups_val, cutoff_real)
    print("Optimized MAP: ", optimized_map)

    print(f"Improvement: {optimized_map - baseline_map}")

    results = pd.DataFrame(trials.results)
    path = '../result_experiments/XGBoostBayesianOptimizationResults.csv'
    results.to_csv(path, index=False)


if __name__ == '__main__':
    __main__()
