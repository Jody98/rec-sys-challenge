import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.model_selection import train_test_split


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


def evaluate_algorithm(URM_test, recommender_object, at=10):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_AP = 0.0

    num_eval = 0

    URM_test = sps.csr_matrix(URM_test)

    for user_id in range(URM_test.shape[0]):
        relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(user_id, cutoff=at)

            num_eval += 1
            cumulative_precision += precision(recommended_items, relevant_items)
            cumulative_recall += recall(recommended_items, relevant_items)
            cumulative_AP += average_precision(recommended_items, relevant_items)

    cumulative_precision /= num_eval
    cumulative_recall /= num_eval
    MAP = cumulative_AP / num_eval

    print("Recommender performance is: Precision = {:.7f}, Recall = {:.7f}, MAP = {:.7f}".format(cumulative_precision,
                                                                                                 cumulative_recall,
                                                                                                 MAP))

    return cumulative_precision, cumulative_recall, MAP


def preprocess_data(ratings: pd.DataFrame, df1: pd.DataFrame):
    unique_users = np.unique(np.concatenate([ratings.row.unique(), df1.user_id.unique()]))
    unique_items = ratings.col.unique()

    num_users, min_user_id, max_user_id = unique_users.size, unique_users.min(), unique_users.max()
    num_items, min_item_id, max_item_id = unique_items.size, unique_items.min(), unique_items.max()

    print(num_users, min_user_id, max_user_id)
    print(num_items, min_item_id, max_item_id)

    mapping_user_id = pd.DataFrame({"mapped_user_id": np.arange(num_users), "row": unique_users})
    mapping_item_id = pd.DataFrame({"mapped_item_id": np.arange(num_items), "col": unique_items})

    ratings = pd.merge(left=ratings, right=mapping_user_id, how="inner", on="row")
    ratings = pd.merge(left=ratings, right=mapping_item_id, how="inner", on="col")

    df1 = pd.merge(left=df1, right=mapping_user_id, how="inner", left_on="user_id", right_on="row")
    df1.drop(columns=["row"], inplace=True)

    return ratings, df1


def dataset_splits(ratings, num_users, num_items, validation_percentage: float, testing_percentage: float):
    seed = 1234

    (user_ids_training, user_ids_test,
     item_ids_training, item_ids_test,
     ratings_training, ratings_test) = train_test_split(ratings.mapped_user_id,
                                                        ratings.mapped_item_id,
                                                        ratings.data,
                                                        test_size=testing_percentage,
                                                        shuffle=True,
                                                        random_state=seed)

    (user_ids_training, user_ids_validation,
     item_ids_training, item_ids_validation,
     ratings_training, ratings_validation) = train_test_split(user_ids_training,
                                                              item_ids_training,
                                                              ratings_training,
                                                              test_size=validation_percentage,
                                                              )

    URM_train = sps.csr_matrix((ratings_training, (user_ids_training, item_ids_training)),
                               shape=(num_users, num_items))

    URM_validation = sps.csr_matrix((ratings_validation, (user_ids_validation, item_ids_validation)),
                                    shape=(num_users, num_items))

    URM_test = sps.csr_matrix((ratings_test, (user_ids_test, item_ids_test)),
                              shape=(num_users, num_items))

    URM_all = URM_train + URM_validation + URM_test
    URM_train_validation = URM_train + URM_validation

    return URM_train, URM_validation, URM_test, URM_all, URM_train_validation


def prepare_submission(df1: pd.DataFrame, ratings: pd.DataFrame, recommender, mapping_to_item_id):
    users_ids_and_mappings = df1[["user_id", "mapped_user_id"]].drop_duplicates()
    items_ids_and_mappings = ratings[["col", "mapped_item_id"]].drop_duplicates()

    recommendation_length = 10
    submission = []
    for idx, row in users_ids_and_mappings.iterrows():
        user_id = row.user_id
        mapped_user_id = row.mapped_user_id

        recommendations = recommender.recommend(mapped_user_id, cutoff=recommendation_length)

        submission.append((user_id, [mapping_to_item_id[item_id] for item_id in recommendations]))

    return submission


def write_submission(submissions):
    with open("./sample_submission.csv", "w") as f:
        f.write("user_id,item_list\n")
        for user_id, items in submissions:
            f.write(f"{user_id},{' '.join([str(item) for item in items])}\n")


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


def read_data(data_file_path, users_file_path):
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
