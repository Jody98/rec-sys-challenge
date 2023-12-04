import numpy as np
import pandas as pd
import scipy.sparse as sps
from sklearn.model_selection import train_test_split

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.KNN import ItemKNNCFRecommender
from challenge.utils.functions import read_data, evaluate_algorithm


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    dataframe1 = pd.read_csv(users_file_path)
    dataframe2 = pd.read_csv(data_file_path)

    dataframe2, dataframe1 = preprocess_data(dataframe2, dataframe1)

    mapping_to_item_id = dict(zip(dataframe2.mapped_item_id, dataframe2.col))

    URM_train, URM_validation, URM_test, URM_all, URM_train_validation = dataset_splits(dataframe2,
                                                                                        num_users=12859,
                                                                                        num_items=22222,
                                                                                        validation_percentage=0.20,
                                                                                        testing_percentage=0.20)

    recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train_validation)
    recommender.fit(topK=10, shrink=19, similarity='jaccard', normalize=False, feature_weighting="TF-IDF")

    submission = prepare_submission(dataframe1, dataframe2, recommender, mapping_to_item_id)
    write_submission(submission)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    evaluate_algorithm(URM_test, recommender, cutoff_list[0])


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


if __name__ == '__main__':
    __main__()
