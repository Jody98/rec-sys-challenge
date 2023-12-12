import numpy as np
import pandas as pd
import scipy.sparse as sps
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import plot_importance

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import P3alphaRecommender, RP3betaRecommender
from Recommenders.Hybrid.GeneralizedLinearHybridRecommender import GeneralizedLinearHybridRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.NonPersonalizedRecommender import TopPop
from Recommenders.SLIM import SLIMElasticNetRecommender
from Recommenders.EASE_R import EASE_R_Recommender
from Recommenders.MatrixFactorization import IALSRecommender
from challenge.utils.functions import read_data


def model_training(X, y, n_estimators, max_depth, gamma, reg_lambda, reg_alpha, booster, learning_rate=0.01):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False,
                              enable_categorical=True,
                              eval_metric='logloss',
                              early_stopping_rounds=10,
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

    pred_labels_tr = model.predict(X_train)
    pred_labels_te = model.predict(X_test)

    print("Training predictions: ", pred_labels_tr)
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
    print("")

    print('*************** Evaluation on Training Data ***************')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    print(classification_report(y_train, pred_labels_tr, zero_division=1))
    print('--------------------------------------------------------')

    return clf, X_test, y_test


def fine_tune_xgboost(X, y):
    param_grid = {
        'n_estimators': [80, 100, 125],  # da aumentare
        'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1],  # da diminuire
        'max_depth': [1, 3, 5],  # da diminuire
        'gamma': [0.1, 0.5],
        'reg_lambda': [1e-7, 1e-6, 1e-5],  # da diminuire
        'reg_alpha': [6, 7, 8],  # da aumentare
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
    submission_file_path = '../output_files/XGBoostSubmission.csv'
    submission_file_path_reranked = '../output_files/XGBoostSubmissionReranked.csv'
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'

    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    evaluator = EvaluatorHoldout(URM_validation, cutoff_list=[10])

    n_users, n_items = URM_all.shape

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=9, shrink=19, similarity='tversky', tversky_alpha=0.036424892090848766,
                         tversky_beta=0.9961018325655608)

    results, _ = evaluator.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender MAP: {}".format(results.loc[10]["MAP"]))

    rp3beta = RP3betaRecommender.RP3betaRecommender(URM_train)
    rp3beta.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                implicit=True, normalize_similarity=True)
    RP3_Wsparse = rp3beta.W_sparse

    results, _ = evaluator.evaluateRecommender(rp3beta)
    print("RP3betaRecommender MAP: {}".format(results.loc[10]["MAP"]))

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_all)
    SLIM_recommender.fit(topK=216, l1_ratio=0.0032465600313226354, alpha=0.002589066655986645, positive_only=True)

    results, _ = evaluator.evaluateRecommender(SLIM_recommender)
    print("SLIMElasticNetRecommender MAP: {}".format(results.loc[10]["MAP"]))

    recommenders = [item_recommender, item_recommender, item_recommender, rp3beta, SLIM_recommender]
    gamma = 0.42759799127984477
    delta = 4.3291270788055805
    epsilon = 4.657898008053695

    hybrid = GeneralizedLinearHybridRecommender(URM_train, recommenders=recommenders)
    hybrid.fit(gamma=gamma, delta=delta, epsilon=epsilon)

    results, _ = evaluator.evaluateRecommender(hybrid)
    print("GeneralizedLinearHybridRecommender MAP: {}".format(results.loc[10]["MAP"]))

    training_dataframe = pd.DataFrame(index=range(0, n_users), columns=["ItemID"])
    training_dataframe.index.name = 'UserID'

    user_recommendations_items = []
    user_recommendations_user_id = []

    for user_id in tqdm(range(n_users)):
        recommendations = hybrid.recommend(user_id, cutoff=cutoff_xgb)
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

    topPop = TopPop(URM_train)
    topPop.fit()

    results, _ = evaluator.evaluateRecommender(topPop)
    print("TopPop MAP: {}".format(results.loc[10]["MAP"]))

    p3alpha = P3alphaRecommender.P3alphaRecommender(URM_train)
    p3alpha.fit(topK=40, alpha=0.3119217553589628, min_rating=0.01, implicit=True,
                normalize_similarity=True)

    results, _ = evaluator.evaluateRecommender(p3alpha)
    print("P3alphaRecommender MAP: {}".format(results.loc[10]["MAP"]))

    EASE_R = EASE_R_Recommender.EASE_R_Recommender(URM_train)
    EASE_R.fit(topK=59, l2_norm=29.792347118106623, normalize_matrix=False)
    EASE_R_Wsparse = sps.csr_matrix(EASE_R.W_sparse)

    results, _ = evaluator.evaluateRecommender(EASE_R)
    print("EASE_R_Recommender MAP: {}".format(results.loc[10]["MAP"]))

    ials_recommender = IALSRecommender.IALSRecommender(URM_train)
    ials_recommender.fit(epochs=10, num_factors=92, confidence_scaling="linear", alpha=2.5431444656816597,
                         epsilon=0.035779451402656745,
                         reg=1.5, init_mean=0.0, init_std=0.1)

    results, _ = evaluator.evaluateRecommender(ials_recommender)
    print("IALSRecommender MAP: {}".format(results.loc[10]["MAP"]))

    other_algorithms = {
        "TopPop": topPop,
        "ItemKNNCF": item_recommender,
        "P3alpha": p3alpha,
        "RP3beta": rp3beta,
        "SLIM": SLIM_recommender,
        "EASE_R": EASE_R,
        "IALS": ials_recommender
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
    y_pred2 = dTree_clf.predict(X_test)

    print("Accuracy before boosting:", accuracy_score(y_test, y_pred2))
    print("Precision before boosting:", precision_score(y_test, y_pred2))
    print("Recall before boosting:", recall_score(y_test, y_pred2))
    print("F1 before boosting:", f1_score(y_test, y_pred2))

    clf, X_test, y_test = model_training(X, y, **xgboost_grid_search.best_params_)

    plot1 = plot_importance(clf, importance_type='gain', title='Gain')
    plot2 = plot_importance(clf, importance_type='cover', title='Cover')
    plot3 = plot_importance(clf, importance_type='weight', title='Weight (Frequence)')

    plot1.figure.savefig('gain.png')
    plot2.figure.savefig('cover.png')
    plot3.figure.savefig('weight.png')

    model = xgb.XGBClassifier(use_label_encoder=False,
                              enable_categorical=True,
                              eval_metric='logloss',
                              early_stopping_rounds=10,
                              **xgboost_grid_search.best_params_
                              )

    eval_set = [(X_val, y_val)]
    hybrid_all = model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True
    )
    predictions = model.predict(X_test)

    print("Accuracy after boosting:", accuracy_score(y_test, predictions))
    print("Precision after boosting:", precision_score(y_test, predictions))
    print("Recall after boosting:", recall_score(y_test, predictions))
    print("F1 after boosting:", f1_score(y_test, predictions))

    '''training_dataframe["Recommendation"] = pd.Series(predictions, index=training_dataframe.index)
    reranked_df = training_dataframe.sort_values(by=['UserID', 'Recommendation'], ascending=[True, False])

    write_recommendations(item_recommender, submission_file_path, users_list, cutoff=cutoff_real)
    write_reranked_recommendations(submission_file_path_reranked, users_list, cutoff=cutoff_real,
                                   reranked_df=reranked_df)'''


if __name__ == '__main__':
    __main__()
