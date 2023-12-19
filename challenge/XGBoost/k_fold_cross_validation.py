import numpy as np
import pandas as pd
import scipy.sparse as sps
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)
    URM_train, URM_validation = split_train_in_two_percentage_global_sample(URM_train, train_percentage=0.80)

    URM_train_recommenders, URM_train_booster = train_test_split(URM_train, test_size=0.5, random_state=42)

    param_grid = {
        'n_estimators': [50],
        'learning_rate': [1e-2],
        'reg_alpha': [1],
        'reg_lambda': [1],
        'max_depth': [5],
        'max_leaves': [5],
        'grow_policy': ['depthwise'],
        'objective': ['binary:logistic'],
        'booster': ['gbtree'],
        'enable_categorical': [True],
    }

    n_users, n_items = URM_train.shape

    rp3beta = RP3betaRecommender.RP3betaRecommender(URM_train)
    rp3beta.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                implicit=True, normalize_similarity=True)
    RP3_Wsparse = rp3beta.W_sparse

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

    groups = training_dataframe.groupby("UserID").size().values

    objective = 'pairwise'
    model = xgb.XGBRanker(objective='rank:{}'.format(objective), enable_categorical=True)

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, error_score='raise')
    grid_search.fit(X_train, y_train, group=groups, verbose=True)

    print("Migliori parametri: ", grid_search.best_params_)
    print("Miglior score: ", grid_search.best_score_)
    print("Miglior modello: ", grid_search.best_estimator_)

    results = pd.DataFrame(grid_search.cv_results_)
    path = '../result_experiments/XGBoostGridSearchResults.csv'
    results.to_csv(path, index=False)


def model_training(X, y, n_trees, mdepth, gamma, lam):
    ##### Step 1 - Create training and testing samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ##### Step 2 - Set model and its parameters
    model = xgb.XGBClassifier(use_label_encoder=False,
                              booster='gbtree',  # boosting algorithm to use, default gbtree, othera: gblinear, dart
                              n_estimators=n_trees,  # number of trees, default = 100
                              eta=0.3,  # this is learning rate, default = 0.3
                              max_depth=mdepth,  # maximum depth of the tree, default = 6
                              gamma=gamma,  # used for pruning, if gain < gamma the branch will be pruned, default = 0
                              reg_lambda=lam,  # regularization parameter, defautl = 1
                              # min_child_weight=0 # this refers to Cover which is also responsible for pruning if not set to 0
                              )

    # Fit the model
    clf = model.fit(X_train, y_train)

    ##### Step 3
    # Predict class labels on training data
    pred_labels_tr = model.predict(X_train)
    # Predict class labels on a test data
    pred_labels_te = model.predict(X_test)

    print("Training predictions: ", pred_labels_tr)
    print("Testing predictions: ", pred_labels_te)

    ##### Step 4 - Model summary
    # Basic info about the model
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
    # Look at classification report to evaluate the model
    print(classification_report(y_test, pred_labels_te))
    print('--------------------------------------------------------')
    print("")

    print('*************** Evaluation on Training Data ***************')
    score_tr = model.score(X_train, y_train)
    print('Accuracy Score: ', score_tr)
    # Look at classification report to evaluate the model
    print(classification_report(y_train, pred_labels_tr))
    print('--------------------------------------------------------')

    return clf, X_test, y_test


if __name__ == '__main__':
    __main__()
