import numpy as np
import scipy.sparse as sps
import pandas as pd


def precision(recommended_items, relevant_items):
    is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)
    precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)

    return precision_score


def recall(recommended_items, relevant_items):
    is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)
    recall_score = np.sum(is_relevant, dtype=np.float32) / relevant_items.shape[0]

    return recall_score


def average_precision(recommended_items, relevant_items):
    is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)
    precision_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
    average_precision_score = np.sum(precision_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
    return average_precision_score


def evaluate_algorithm(URM_test, recommender_object, at=5):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_AP = 0.0

    num_eval = 0

    URM_test = sps.csr_matrix(URM_test)

    for user_id in range(URM_test.shape[0]):
        relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(user_id, at=at)[1]

            num_eval += 1
            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_AP += average_precision(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    MAP = cumulative_AP / num_eval

    print("Recommender performance is: Precision = {:.4f}, Recall = {:.4f}, MAP = {:.4f}".format(cumulative_precision,
                                                                                                 cumulative_recall,
                                                                                                 MAP))


def generate_submission_csv(filename, recommendations):
    with open(filename, 'w') as f:
        f.write("user_id,item_list\n")
        for recommendation in recommendations:
            f.write("{},{}\n".format(recommendation["user_id"],
                                     " ".join(str(x) for x in recommendation["item_list"])))


def split(URM_all):
    train_test_split = 0.8
    n_interactions = URM_all.nnz
    train_mask = np.random.choice([True, False], n_interactions, p=[train_test_split, 1 - train_test_split])

    URM_all = URM_all.tocoo()
    URM_train = sps.csr_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))

    test_mask = np.logical_not(train_mask)
    URM_test = sps.csr_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))
    return URM_train, URM_test


def read_data():
    data_file_path = 'input_files/data_train.csv'
    users_file_path = 'input_files/data_target_users_test.csv'
    URM_all_dataframe = pd.read_csv(filepath_or_buffer=data_file_path,
                                    sep=',',
                                    engine='python')
    URM_all_dataframe.columns = ['UserID', 'ItemID', 'Data']

    test_users = pd.read_csv(filepath_or_buffer=users_file_path,
                             sep=',',
                             engine='python')
    test_users.columns = ['UserID']
    users_list = test_users['UserID'].values
    return URM_all_dataframe, users_list
