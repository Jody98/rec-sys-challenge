import numpy as np
import pandas as pd
import scipy.sparse as sps
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
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


def diversity(items_interacted):
    unique_items = set(items_interacted)
    return len(unique_items) / len(items_interacted) if items_interacted else 0


def model_training(X, y, n_estimators, max_depth, gamma, reg_lambda, reg_alpha, booster, learning_rate=0.01):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False,
                              enable_categorical=True,
                              eval_metric='logloss',
                              early_stopping_rounds=20,
                              booster=booster,
                              n_estimators=n_estimators,
                              max_depth=max_depth,
                              gamma=gamma,
                              reg_lambda=reg_lambda,
                              learning_rate=learning_rate,
                              reg_alpha=reg_alpha,
                              )

    eval_set = [(X_val, y_val)]
    clf = model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True
    )

    pred_labels_te = model.predict(X_test)

    print("Testing predictions: ", pred_labels_te)

    print('*************** Tree Summary ***************')
    print('No. of classes: ', clf.n_classes_)
    print('Classes: ', clf.classes_)
    print('No. of features: ', clf.n_features_in_)
    print('No. of Estimators: ', clf.n_estimators)
    print('--------------------------------------------------------')
    print("")

    print('*************** Evaluation on Test Data ***************')
    score_te = model.score(X_test, y_test)
    print('Accuracy Score: ', score_te)
    print(classification_report(y_test, pred_labels_te, zero_division=1))
    print('--------------------------------------------------------')

    return clf, X_test, y_test


def fine_tune_xgboost(X, y):
    param_grid = {
        'n_estimators': [250, 350, 500],  # da aumentare
        'learning_rate': [0.0001, 0.001, 0.01],  # da diminuire
        'max_depth': [1, 2],  # da diminuire
        'gamma': [0.05, 0.15, 0.25],  # da diminuire
        'reg_lambda': [1e-7, 1e-5, 1e-3],  # da diminuire
        'reg_alpha': [4, 5, 7],  # da aumentare
        'booster': ['gbtree'],
    }

    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, enable_categorical=True)
    grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, scoring='accuracy', cv=10)
    grid_search.fit(X, y)

    print("Parametri ottimali della Grid Search:", grid_search.best_params_)

    return grid_search


def write_recommendations(recommender, file_name, users_list, cutoff=10):
    recommendations = 'user_id,item_list'
    f = open(file_name, "w")

    for user_id in users_list:
        recommendations_per_user = recommender.recommend(user_id_array=user_id, remove_seen_flag=True, cutoff=cutoff)

        recommendation_string = str(user_id) + ','

        for rec in recommendations_per_user:
            recommendation_string = recommendation_string + str(rec) + ' '

        recommendation_string = recommendation_string[:-1]
        recommendations = recommendations + '\n' + recommendation_string

    f.write(recommendations)
    f.close()


def write_reranked_recommendations(file_name, users_list, cutoff=10, reranked_df=None):
    recommendations = 'user_id,item_list'
    f = open(file_name, "w")

    for user_id in users_list:
        recommendations_per_user = reranked_df.loc[reranked_df['UserID'] == user_id]['ItemID'].values[:cutoff]

        recommendation_string = str(user_id) + ','

        for rec in recommendations_per_user:
            recommendation_string = recommendation_string + str(rec) + ' '

        recommendation_string = recommendation_string[:-1]
        recommendations = recommendations + '\n' + recommendation_string

    f.write(recommendations)
    f.close()


def __main__():
    cutoff_real = 10
    cutoff_xgb = 20
    cutoff_list = [10]
    folder_path = "../result_experiments/"
    EASE80 = "EASE_R_Recommender_best_model80.zip"
    SLIM80 = "SLIMElasticNetRecommender_best_model80.zip"
    MultVAE80 = "MultVAERecommender_best_model80.zip"
    ALS80 = "ALSRecommender_best_model80.zip"
    IALS80 = "IALSRecommender_best_model80.zip"
    submission_file_path = '../output_files/XGBoostSubmission.csv'
    submission_file_path_reranked = '../output_files/XGBoostSubmissionReranked.csv'
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'

    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz('../input_files/URM_train_plus_validation.npz')
    URM_validation = sps.load_npz('../input_files/URM_test.npz')
    URM_all = sps.load_npz('../input_files/URM_all.npz')

    evaluator = EvaluatorHoldout(URM_validation, cutoff_list=cutoff_list)

    n_users, n_items = URM_all.shape

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
        "EASE_R": EASE_R,
        "Hybrid": RP3_recommender,
        "SLIM": SLIM_recommender,
        "Item": item_recommender
    }

    all_recommender = HybridLinear(URM_train, recommenders)
    all_recommender.fit(MultVAE=12, ALS=-0.38442274063330273, EASE_R=0,
                        Hybrid=1.7016467362004866, SLIM=1.9523771152704132, Item=0.5974437062485289)

    results, _ = evaluator.evaluateRecommender(all_recommender)
    print("HybridLinear")
    print("MAP: {}".format(results.loc[10]["MAP"]))

    training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    training_dataframe.index.name = 'UserID'

    user_recommendations_items = []
    user_recommendations_user_id = []

    for user_id in tqdm(range(n_users)):
        recommendations = all_recommender.recommend(user_id, cutoff=cutoff_xgb)
        training_dataframe.loc[user_id, "ItemID"] = recommendations
        user_recommendations_items.extend(recommendations)
        user_recommendations_user_id.extend([id] * len(recommendations))

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

    other_algorithms = {
        "TopPop": topPop,
        "UserKNNCF": User,
        "ItemKNNCF": item_recommender,
        "P3alpha": P3_recommender,
        "RP3beta": RP3_recommender,
        "SLIM": SLIM_recommender,
        "EASE_R": EASE_R,
        "IALS": IALS,
        "PureSVD": pureSVD,
        "PureSVDItem": pureSVDitem,
        "ALS": ALS,
        "MultVAE": MultVAE,
        "HybridKNN": hybrid_recommender
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
    item_interaction_count = np.diff(sps.csc_matrix(URM_all).indptr)

    training_dataframe['avg_user_interaction'] = training_dataframe['user_profile_len'] / n_users
    training_dataframe['avg_item_interaction'] = training_dataframe['item_popularity'] / n_items

    total_interactions = URM_all.nnz
    training_dataframe['user_interaction_ratio'] = training_dataframe['user_profile_len'] / total_interactions
    training_dataframe['item_interaction_ratio'] = training_dataframe['item_popularity'] / total_interactions

    URM_csr = sps.csr_matrix(URM_all)
    user_diversity = URM_csr.copy()
    user_diversity.data = np.ones_like(user_diversity.data)
    user_diversity = np.array(user_diversity.sum(axis=1)).squeeze() / user_interaction_count
    training_dataframe['user_diversity'] = user_diversity[training_dataframe["UserID"].values.astype(int)]

    max_user_interactions = user_interaction_count.max()
    training_dataframe['normalized_user_interaction'] = training_dataframe[
                                                            'user_profile_len'] / max_user_interactions

    max_item_interactions = item_interaction_count.max()
    training_dataframe['normalized_item_interaction'] = training_dataframe[
                                                            'item_popularity'] / max_item_interactions

    kmeans = KMeans(n_clusters=10, random_state=42).fit(URM_all.T)
    user_clusters = kmeans.labels_
    training_dataframe['user_cluster'] = [user_clusters[user] for user in training_dataframe['UserID']]

    URM_lil = URM_all.tolil()
    training_dataframe['diversity'] = [diversity(URM_lil.rows[user]) for user in training_dataframe['UserID']]

    training_dataframe['Recommendation'] = pd.Series(target, index=training_dataframe.index)

    y = training_dataframe["Recommendation"]
    X = training_dataframe.drop(columns=["Label", "Recommendation"])
    X["UserID"] = X["UserID"].astype("category")
    X["ItemID"] = X["ItemID"].astype("category")

    xgboost_grid_search = fine_tune_xgboost(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    dTree_clf = DecisionTreeClassifier()

    dTree_clf.fit(X_train, y_train)
    predictions = dTree_clf.predict(X_test)

    print("Accuracy before boosting:", accuracy_score(y_test, predictions))
    print("Precision before boosting:", precision_score(y_test, predictions))
    print("Recall before boosting:", recall_score(y_test, predictions))
    print("F1 before boosting:", f1_score(y_test, predictions))

    clf, X_test, y_test = model_training(X, y, **xgboost_grid_search.best_params_)

    plot1 = plot_importance(clf, importance_type='gain', title='Gain')
    plot2 = plot_importance(clf, importance_type='cover', title='Cover')
    plot3 = plot_importance(clf, importance_type='weight', title='Weight')

    plot1.figure.savefig('gain.png')
    plot2.figure.savefig('cover.png')
    plot3.figure.savefig('weight.png')

    model = xgb.XGBClassifier(use_label_encoder=False,
                              enable_categorical=True,
                              eval_metric='logloss',
                              early_stopping_rounds=20,
                              **xgboost_grid_search.best_params_
                              )

    eval_set = [(X_val, y_val)]
    hybrid_all = model.fit(
        X_train, y_train,  # mettere X_train, y_train
        eval_set=eval_set,
        verbose=True
    )
    predictions = model.predict(X_test) # mettere X_test

    print("Accuracy after boosting:", accuracy_score(y_test, predictions)) # mettere y_test
    print("Precision after boosting:", precision_score(y_test, predictions))
    print("Recall after boosting:", recall_score(y_test, predictions))
    print("F1 after boosting:", f1_score(y_test, predictions))

    training_dataframe["Recommendation"] = pd.Series(predictions, index=training_dataframe.index)
    reranked_df = training_dataframe.sort_values(by=['UserID', 'Recommendation'], ascending=[True, False])

    write_recommendations(item_recommender, submission_file_path, users_list, cutoff=cutoff_real)
    write_reranked_recommendations(submission_file_path_reranked, users_list, cutoff=cutoff_real,
                                   reranked_df=reranked_df)


if __name__ == '__main__':
    __main__()
