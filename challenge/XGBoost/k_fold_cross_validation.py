import numpy as np
import pandas as pd
import scipy.sparse as sps
import xgboost as xgb
from hyperopt import fmin, tpe, Trials, STATUS_OK, hp
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from tqdm import tqdm
from xgboost import plot_importance

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import P3alphaRecommender, RP3betaRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from challenge.utils.functions import read_data


def cross_val_score_model(X, y, groups_fitting, params, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    map_scores = []

    groups = X['UserID'].values

    for train_index, test_index in gkf.split(X, y, groups=groups):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]

        model = xgb.XGBRanker(objective='rank:pairwise', **params, enable_categorical=True, booster='gbtree')
        model.fit(X_train, y_train, group=groups_fitting[:10420], verbose=False)

        recommendations = []
        relevancies = []
        unique_users = X_val['UserID'].unique()
        for user_id in unique_users:
            user_data = X_val[X_val['UserID'] == user_id]
            user_items = user_data['ItemID'].values
            user_true_item = y_val[X_val['UserID'] == user_id].values
            user_pred = model.predict(user_data)

            recommended_items = user_items[np.argsort(user_pred)[::-1]]
            recommendations.append(recommended_items)
            relevancies.append(user_true_item)

        # Calcola MAP per il fold corrente
        map_score = mean_average_precision(recommendations, relevancies, k=10)
        map_scores.append(map_score)

    return np.mean(map_scores)


def cross_val_score_modelv2(X, y, groups, params, n_splits=5):
    cutoff_real = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    map_scores = []

    for train_index, test_index in kf.split(X, groups=groups):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        groups_train, groups_val = groups[train_index], groups[test_index]

        model = xgb.XGBRanker(objective='rank:pairwise', **params, enable_categorical=True, booster='gbtree')
        model.fit(X_train, y_train, group=groups_train, verbose=False)

        # Preparare le raccomandazioni e le rilevanze
        recommendations = []
        relevancies = []
        unique_users = X_val['UserID'].unique()
        for user_id in unique_users:
            user_data = X_val[X_val['UserID'] == user_id]
            user_items = user_data['ItemID'].values
            user_true_item = y_val[X_val['UserID'] == user_id].values
            user_pred = model.predict(user_data)

            # Ordina gli items per il punteggio predetto
            recommended_items = user_items[np.argsort(user_pred)[::-1]]
            recommendations.append(recommended_items)
            relevancies.append(user_true_item)

        # Calcola MAP per il fold corrente
        map_score = mean_average_precision(recommendations, relevancies, k=cutoff_real)
        map_scores.append(map_score)

    return np.mean(map_scores)


def precision_at_k(recommended_list, relevant_list, k):
    if len(recommended_list) > k:
        recommended_list = recommended_list[:k]

    num_hits = len(set(recommended_list) & set(relevant_list))
    return num_hits / k


def recall_at_k(recommended_list, relevant_list, k):
    if len(recommended_list) > k:
        recommended_list = recommended_list[:k]

    num_hits = len(set(recommended_list) & set(relevant_list))
    return num_hits / len(relevant_list) if len(relevant_list) > 0 else 0


def avg_precision(recommended_list, relevant_list):
    ap_sum = 0.0
    num_hits = 0.0

    for i, p in enumerate(recommended_list):
        if p in relevant_list and p not in recommended_list[:i]:
            num_hits += 1.0
            ap_sum += num_hits / (i + 1.0)

    if len(relevant_list) > 0:
        return ap_sum / len(relevant_list)
    else:
        return 0.0


def mean_average_precision(recommendations, relevancies, k):
    return np.mean([avg_precision(rec[:k], rel) for rec, rel in zip(recommendations, relevancies)])


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
    k = 10
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

    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    evaluator = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)

    space = {
        'n_estimators': hp.choice('n_estimators', [250, 400, 500]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'reg_alpha': hp.uniform('reg_alpha', 0, 5),
        'reg_lambda': hp.uniform('reg_lambda', 0, 5),
        'max_depth': hp.choice('max_depth', [5, 10, 15]),
        'max_leaves': hp.choice('max_leaves', [0, 5, 10]),
        'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),
    }

    n_estimators_choices = [250, 400, 500]
    max_depth_choices = [5, 10, 15]
    max_leaves_choices = [0, 5, 10]
    grow_policy_choices = ['depthwise', 'lossguide']

    n_users, n_items = URM_train.shape

    relevancies = []
    for user_id in range(n_users):
        start_pos = URM_validation.indptr[user_id]
        end_pos = URM_validation.indptr[user_id + 1]

        relevant_items = URM_validation.indices[start_pos:end_pos]
        relevancies.append(relevant_items)

    rp3beta = RP3betaRecommender.RP3betaRecommender(URM_train)
    rp3beta.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                implicit=True, normalize_similarity=True)
    RP3_Wsparse = rp3beta.W_sparse

    results, _ = evaluator.evaluateRecommender(rp3beta)
    print("RP3betaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    training_dataframe.index.name = 'UserID'

    recommendations = []
    for user_id in tqdm(range(n_users)):
        recommended_items = rp3beta.recommend(user_id, cutoff=cutoff_xgb)
        training_dataframe.loc[user_id, "ItemID"] = recommended_items
        recommendations.append(recommended_items)

    baseline_map = mean_average_precision(recommendations, relevancies, k)
    p_at_k = np.mean([precision_at_k(rec, rel, k) for rec, rel in zip(recommendations, relevancies)])
    r_at_k = np.mean([recall_at_k(rec, rel, k) for rec, rel in zip(recommendations, relevancies)])

    print(f"MAP: {baseline_map}, P@K: {p_at_k}, R@K: {r_at_k}")

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

    groups = X_train.groupby("UserID").size().values

    '''def obj(params):
        model = xgb.XGBRanker(objective='rank:pairwise', **params, enable_categorical=True, booster='gbtree')
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

        return {'loss': -score, 'status': STATUS_OK}'''

    def obj(params):
        score = cross_val_score_model(X_train, y_train, groups, params)
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_indices = fmin(fn=obj, space=space, algo=tpe.suggest, max_evals=2, trials=trials)

    best_params = {
        'n_estimators': n_estimators_choices[best_indices['n_estimators']],
        'learning_rate': best_indices['learning_rate'],
        'reg_alpha': best_indices['reg_alpha'],
        'reg_lambda': best_indices['reg_lambda'],
        'max_depth': max_depth_choices[best_indices['max_depth']],
        'max_leaves': max_leaves_choices[best_indices['max_leaves']],
        'grow_policy': grow_policy_choices[best_indices['grow_policy']],
    }

    print("Best Hyperparameters: ", best_params)

    model_optimized = xgb.XGBRanker(objective='rank:pairwise', **best_params, enable_categorical=True, booster='gbtree')
    model_optimized.fit(X_train, y_train, group=groups, verbose=True)

    results = pd.DataFrame(trials.results)
    path = '../result_experiments/XGBoostBayesianOptimizationResults.csv'
    results.to_csv(path, index=False)

    reranked_df = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    reranked_df.index.name = 'UserID'

    recommendations = []
    for user_id in tqdm(range(n_users)):
        X_to_predict = X_train[X_train["UserID"] == user_id]
        X_prediction = model_optimized.predict(X_to_predict)
        dict_prediction = dict(zip(X_to_predict["ItemID"], X_prediction))
        dict_prediction = {k: v for k, v in sorted(dict_prediction.items(), key=lambda item: item[1], reverse=True)}
        list_prediction = list(dict_prediction.keys())[:cutoff_real]
        reranked_df.loc[user_id, "ItemID"] = list_prediction
        recommendations.append(list_prediction)

    optimized_map = mean_average_precision(recommendations, relevancies, k)
    p_at_k = np.mean([precision_at_k(rec, rel, k) for rec, rel in zip(recommendations, relevancies)])
    r_at_k = np.mean([recall_at_k(rec, rel, k) for rec, rel in zip(recommendations, relevancies)])

    print(f"MAP: {optimized_map}, P@K: {p_at_k}, R@K: {r_at_k}")
    print(f"Improvement: {optimized_map - baseline_map}")

    reranked_df['UserID'] = reranked_df.index

    with open(submission_file_path, 'w') as file:
        file.write('user_id,item_list\n')
        for user_id in tqdm(users_list):
            item_list = reranked_df.loc[user_id, "ItemID"]
            user_string = f"{user_id},{' '.join(map(str, item_list))}\n"
            file.write(user_string)

    plot1 = plot_importance(model_optimized, importance_type='gain', title='Gain')
    plot2 = plot_importance(model_optimized, importance_type='cover', title='Cover')
    plot3 = plot_importance(model_optimized, importance_type='weight', title='Weight')

    plot1.figure.savefig('gain.png')
    plot2.figure.savefig('cover.png')
    plot3.figure.savefig('weight.png')


if __name__ == '__main__':
    __main__()
