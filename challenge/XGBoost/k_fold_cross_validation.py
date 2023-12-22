import numpy as np
import pandas as pd
import scipy.sparse as sps
import xgboost as xgb
from hyperopt import hp, Trials, fmin, STATUS_OK, tpe
from sklearn.model_selection import GroupKFold, train_test_split
from tqdm import tqdm
from xgboost import plot_importance

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.GraphBased import P3alphaRecommender, RP3betaRecommender
from Recommenders.Hybrid.HybridLinear import HybridLinear
from Recommenders.KNN import ItemKNNCFRecommender, UserKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityTripleHybridRecommender import ItemKNNSimilarityTripleHybridRecommender
from Recommenders.MatrixFactorization import IALSRecommender, PureSVDRecommender, ALSRecommender
from Recommenders.Neural.MultVAERecommender import MultVAERecommender_PyTorch_OptimizerMask
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data


def get_additional_recommendations(user_id, num_additional_items, URM, popular_items):
    user_interactions = URM[user_id].indices
    additional_recommendations = []

    for item in popular_items:
        if item not in user_interactions and len(additional_recommendations) < num_additional_items:
            additional_recommendations.append(item)

    return additional_recommendations


def get_popular_items(URM):
    item_popularity = np.ediff1d(URM.tocsc().indptr)
    popular_items = np.argsort(item_popularity)[::-1]
    return popular_items


def cross_val_score_model(X, y, groups_fitting, params, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    map_scores = []

    groups = X['UserID'].values

    for train_index, test_index in gkf.split(X, y, groups=groups):
        X_train_full, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train_full, y_val = y.iloc[train_index], y.iloc[test_index]

        X_train, X_internal_val, y_train, y_internal_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42)

        model = xgb.XGBRanker(objective='rank:pairwise', **params, enable_categorical=True, booster='gbtree')

        model.fit(
            X_train, y_train,
            group=groups_fitting[:int(len(X_train)/20)],
            eval_set=[(X_internal_val, y_internal_val)],
            eval_group=[groups_fitting[:int(len(X_internal_val)/20)]],
            verbose=False,
            early_stopping_rounds=20,
            eval_metric='map@10'
        )

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

        map_score = mean_average_precision(recommendations, relevancies, k=10)
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
    folder_path = "../result_experiments/"
    EASE80 = "EASE_R_Recommender_best_model80.zip"
    SLIM80 = "SLIMElasticNetRecommender_best_model80.zip"
    MultVAE80 = "MultVAERecommender_best_model80.zip"
    ALS80 = "ALSRecommender_best_model80.zip"
    IALS80 = "IALSRecommender_best_model80.zip"
    submission_file_path = '../output_files/XGBoostSubmission.csv'
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'

    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_validation = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    evaluator = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)

    space = {
        'n_estimators': hp.choice('n_estimators', [50, 100, 150, 250, 400, 500, 750]),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.00001), np.log(0.1)),
        'reg_alpha': hp.uniform('reg_alpha', 0, 7),
        'reg_lambda': hp.uniform('reg_lambda', 0, 7),
        'max_depth': hp.choice('max_depth', [0, 1, 2, 3, 5, 7, 10]),
        'max_leaves': hp.choice('max_leaves', [0, 1, 2, 3, 5, 7, 10]),
        'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),
    }

    n_estimators_choices = [50, 100, 150, 250, 400, 500, 750]
    max_depth_choices = [0, 1, 2, 3, 5, 7, 10]
    max_leaves_choices = [0, 1, 2, 3, 5, 7, 10]
    grow_policy_choices = ['depthwise', 'lossguide']

    n_users, n_items = URM_all.shape

    relevancies = []
    for user_id in range(n_users):
        start_pos = URM_validation.indptr[user_id]
        end_pos = URM_validation.indptr[user_id + 1]

        relevant_items = URM_validation.indices[start_pos:end_pos]
        relevancies.append(relevant_items)

    topPop = TopPop(URM_train)
    topPop.fit()

    results, _ = evaluator.evaluateRecommender(topPop)
    print("TopPop MAP: {}".format(results.loc[10]["MAP"]))

    User = UserKNNCFRecommender.UserKNNCFRecommender(URM_train)
    User.fit(topK=400, shrink=8, similarity='jaccard', normalize=False, feature_weighting="TF-IDF")

    results, _ = evaluator.evaluateRecommender(User)
    print("UserKNNCFRecommender MAP: {}".format(results.loc[10]["MAP"]))

    pureSVD = PureSVDRecommender.PureSVDRecommender(URM_train)
    pureSVD.fit(num_factors=43)

    results, _ = evaluator.evaluateRecommender(pureSVD)
    print("PureSVD MAP: {}".format(results.loc[10]["MAP"]))

    pureSVDitem = PureSVDRecommender.PureSVDItemRecommender(URM_train)
    pureSVDitem.fit(num_factors=145, topK=28)

    results, _ = evaluator.evaluateRecommender(pureSVDitem)
    print("PureSVDItem MAP: {}".format(results.loc[10]["MAP"]))

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=9, shrink=13, similarity='tversky', tversky_alpha=0.03642489209084876,
                         tversky_beta=0.9961018325655608)
    item_Wsparse = item_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    P3_recommender = P3alphaRecommender.P3alphaRecommender(URM_train)
    P3_recommender.fit(topK=40, alpha=0.3119217553589628, min_rating=0.01, implicit=True, normalize_similarity=True)
    p3alpha_Wsparse = P3_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(P3_recommender)
    print("P3alphaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(RP3_recommender)
    print("RP3betaRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))
    print("RECALL: {}".format(results.loc[10]["RECALL"]))

    hybrid_recommender = ItemKNNSimilarityTripleHybridRecommender(URM_train, p3alpha_Wsparse, item_Wsparse, RP3_Wsparse)
    hybrid_recommender.fit(topK=225, alpha=0.4976629488640914, beta=0.13017801200221196)

    results, _ = evaluator.evaluateRecommender(hybrid_recommender)
    print("ItemKNNSimilarityTripleHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    EASE_R = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    EASE_R.load_model(folder_path, EASE80)
    EASE_R_Wsparse = sps.csr_matrix(EASE_R.W_sparse)

    results, _ = evaluator.evaluateRecommender(EASE_R)
    print("EASE_R_Recommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    SLIM_recommender.load_model(folder_path, SLIM80)
    SLIM_Wsparse = SLIM_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIM_recommender)
    print("SLIMElasticNetRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    ALS = ALSRecommender.ALS(URM_train)
    ALS.load_model(folder_path, ALS80)

    results, _ = evaluator.evaluateRecommender(ALS)
    print("ALSRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    MultVAE = MultVAERecommender_PyTorch_OptimizerMask(URM_train)
    MultVAE.load_model(folder_path, MultVAE80)

    results, _ = evaluator.evaluateRecommender(MultVAE)
    print("MultVAE MAP: {}".format(results.loc[10]["MAP"]))

    IALS = IALSRecommender.IALSRecommender(URM_train)
    IALS.load_model(folder_path, IALS80)

    results, _ = evaluator.evaluateRecommender(IALS)
    print("IALSRecommender MAP: {}".format(results.loc[10]["MAP"]))

    recommenders = {
        "MultVAE": MultVAE,
        "ALS": IALS,
        "Hybrid": RP3_recommender,
        "SLIM": SLIM_recommender,
        "Item": item_recommender
    }

    all_recommender = HybridLinear(URM_train, recommenders)
    all_recommender.fit(MultVAE=14.180249222221073, ALS=-0.38442274063330273,
                        Hybrid=2.060407131177933, SLIM=2.945116702486108, Item=0.9737256690221096)

    results, _ = evaluator.evaluateRecommender(all_recommender)
    print("HybridLinear")
    print("MAP: {}".format(results.loc[10]["MAP"]))
    print("RECALL: {}".format(results.loc[10]["RECALL"]))

    training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    training_dataframe.index.name = 'UserID'

    popular_items = get_popular_items(URM_all)

    recommendations = []
    for user_id in tqdm(range(n_users)):
        recommended_items = all_recommender.recommend(user_id, cutoff=cutoff_xgb)

        if len(recommended_items) < cutoff_xgb:
            num_additional_items = cutoff_xgb - len(recommended_items)
            additional_items = get_additional_recommendations(user_id, num_additional_items, URM_all, popular_items)
            recommended_items.extend(additional_items)

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

    other_algorithms = {
        "TopPop": topPop,
        "UserKNNCF": User,
        "ItemKNNCF": item_recommender,
        "P3alpha": P3_recommender,
        "ALS": ALS,
        "MultVAE": MultVAE,
        "SLIM": SLIM_recommender,
        "PureSVD": pureSVD,
        "PureSVDitem": pureSVDitem,
        "RP3beta": RP3_recommender,
        "IALS": IALS,
        "Hybrid": hybrid_recommender,
        "EASE_R": EASE_R,
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

    user_interaction_count = np.diff(sps.csr_matrix(URM_all).indptr)

    total_interactions = URM_all.nnz
    training_dataframe['user_interaction_ratio'] = training_dataframe['user_profile_len'] / total_interactions
    training_dataframe['item_interaction_ratio'] = training_dataframe['item_popularity'] / total_interactions

    URM_csr = sps.csr_matrix(URM_all)
    user_diversity = URM_csr.copy()
    user_diversity.data = np.ones_like(user_diversity.data)
    user_diversity = np.array(user_diversity.sum(axis=1)).squeeze() / user_interaction_count
    training_dataframe['user_diversity'] = user_diversity[training_dataframe["UserID"].values.astype(int)]

    y = training_dataframe["Label"]
    X = training_dataframe.drop(columns=["Label"])
    X["UserID"] = X["UserID"].astype("category")
    X["ItemID"] = X["ItemID"].astype("category")

    groups = X.groupby("UserID").size().values

    def obj(params):
        score = cross_val_score_model(X, y, groups, params)
        return {'loss': -score, 'status': STATUS_OK}

    trials = Trials()
    best_indices = fmin(fn=obj, space=space, algo=tpe.suggest, max_evals=200, trials=trials)

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

    #best_params = {'n_estimators': 150, 'learning_rate': 0.00013609784874641523, 'reg_alpha': 4.67667151797596,
    #               'reg_lambda': 3.4764904641581107, 'max_depth': 5, 'max_leaves': 0, 'grow_policy': 'lossguide'}

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
    groups_train = groups[:10420]
    groups_val = groups[10420:]

    model_optimized = xgb.XGBRanker(objective='rank:pairwise',
                                    **best_params,
                                    enable_categorical=True,
                                    booster='gbtree',
                                    eval_metric='map@10',
                                    early_stopping_rounds=20,
                                    verbose=True)
    eval_set = [(X_val, y_val)]
    eval_group = [groups_val]
    model_optimized.fit(X_train, y_train, group=groups_train, verbose=True, eval_set=eval_set, eval_group=eval_group)

    reranked_df = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    reranked_df.index.name = 'UserID'

    recommendations = []
    for user_id in tqdm(range(n_users)):
        X_to_predict = X[X["UserID"] == user_id]
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
