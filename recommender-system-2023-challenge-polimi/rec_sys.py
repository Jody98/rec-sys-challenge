import matplotlib.pyplot as pyplot
import numpy as np
import pandas as pd
import scipy.sparse as sps

data_file_path = 'data_train.csv'
users_file_path = 'data_target_users_test.csv'
URM_all_dataframe = pd.read_csv(filepath_or_buffer=data_file_path,
                                sep=',',
                                engine='python')
URM_all_dataframe.columns = ['UserID', 'ItemID', 'Data']
print(URM_all_dataframe.head())

test_users = pd.read_csv(filepath_or_buffer=users_file_path,
                         sep=',',
                         engine='python')
test_users.columns = ['UserID']
print(test_users.head())
users_list = test_users['UserID'].values

print("Number of items\t {}, Number of users\t {}".format(len(URM_all_dataframe['ItemID'].unique()),
                                                          len(URM_all_dataframe['UserID'].unique())))
print("Number of interactions\t {}".format(URM_all_dataframe['Data'].count()))
print("Max value of UserID\t {}, Max value of ItemID\t {}".format(URM_all_dataframe['UserID'].max(),
                                                                  URM_all_dataframe['ItemID'].max()))

mapped_id, original_id = pd.factorize(URM_all_dataframe['UserID'].unique())
user_original_ID_to_index = pd.Series(data=mapped_id, index=original_id)

mapped_id, original_id = pd.factorize(URM_all_dataframe['ItemID'].unique())
item_original_ID_to_index = pd.Series(data=mapped_id, index=original_id)

URM_all_dataframe['UserID'] = URM_all_dataframe['UserID'].map(user_original_ID_to_index)
URM_all_dataframe['ItemID'] = URM_all_dataframe['ItemID'].map(item_original_ID_to_index)

print(URM_all_dataframe.head())

userID_unique = URM_all_dataframe['UserID'].unique()
itemID_unique = URM_all_dataframe['ItemID'].unique()

n_users = len(userID_unique)
n_items = len(itemID_unique)
n_interactions = len(URM_all_dataframe)

print("Number of items\t {}, Number of users\t {}".format(n_items, n_users))
print("Max value of UserID\t {}, Max value of ItemID\t {}".format(userID_unique.max(), itemID_unique.max()))
print("Number of interactions\t {}".format(n_interactions))

print("Average interactions per user {:.2f}\nAverage interactions per item {:.2f}".format(n_interactions / n_users,
                                                                                          n_interactions / n_items))

print("Sparsity {:.5f} %".format((1 - n_interactions / (n_users * n_items)) * 100))

URM_all = sps.coo_matrix(
    (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
URM_all = URM_all.tocsr()

item_popularity = np.ediff1d(URM_all.tocsc().indptr)
item_popularity = np.sort(item_popularity)

pyplot.plot(item_popularity, 'ro')
pyplot.ylabel('Num Interactions ')
pyplot.xlabel('Item Index Sorted by Popularity')
pyplot.show()

ten_percent = int(n_items / 10)

print("Average per-item interactions over the whole dataset {:.2f}\n".format(item_popularity.mean()))
print("Average per-item interactions for the top 10% popular items {:.2f}\n".format(
    item_popularity[-ten_percent:].mean()))
print("Average per-item interactions for the least 10% popular items {:.2f}\n".format(
    item_popularity[:ten_percent].mean()))
print("Average per-item interactions for the median 10% popular items {:.2f}\n".format(
    item_popularity[int(n_items / 2 - ten_percent / 2):int(n_items / 2 + ten_percent / 2)].mean()))

user_activity = np.ediff1d(URM_all.tocsr().indptr)
user_activity = np.sort(user_activity)

pyplot.plot(user_activity, 'ro')
pyplot.ylabel('Num Interactions ')
pyplot.xlabel('User Index Sorted by Popularity')
pyplot.show()

train_test_split = 0.8
n_interactions = URM_all.nnz
train_mask = np.random.choice([True, False], n_interactions, p=[train_test_split, 1 - train_test_split])

URM_all = URM_all.tocoo()
URM_train = sps.csr_matrix((URM_all.data[train_mask], (URM_all.row[train_mask], URM_all.col[train_mask])))

test_mask = np.logical_not(train_mask)
URM_test = sps.csr_matrix((URM_all.data[test_mask], (URM_all.row[test_mask], URM_all.col[test_mask])))

user_id = 124
relevant_items = URM_test[user_id].indices
recommended_items = np.array([241, 1622, 15, 857, 5823])
is_relevant = np.isin(recommended_items, relevant_items, assume_unique=True)


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


class RandomRecommender(object):

    def fit(self, URM_train):
        self.n_items = URM_train.shape[1]

    def recommend(self, user_id, at=10):
        recommended_items = np.random.choice(self.n_items, at)
        recommendation = {"user_id": user_id, "item_list": recommended_items}
        return recommendation

    def generate_submission_csv(self, filename, recommendations):
        with open(filename, 'w') as f:
            f.write("user_id,item_list\n")
            for recommendation in recommendations:
                f.write("{},{}\n".format(recommendation["user_id"],
                                         " ".join(str(x) for x in recommendation["item_list"])))



def evaluate_algorithm(URM_test, recommender_object, at=5):
    cumulative_precision = 0.0
    cumulative_recall = 0.0
    cumulative_AP = 0.0

    num_eval = 0

    URM_test = sps.csr_matrix(URM_test)

    for user_id in range(URM_test.shape[0]):
        relevant_items = URM_test.indices[URM_test.indptr[user_id]:URM_test.indptr[user_id + 1]]

        if len(relevant_items) > 0:
            recommended_items = recommender_object.recommend(user_id, at=at)

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


random_recommender = RandomRecommender()
random_recommender.fit(URM_train)

recommendations = []

for user_id in users_list:
    recommendation = random_recommender.recommend(user_id, at=10)
    recommendations.append(recommendation)

random_recommender.generate_submission_csv("submission_random.csv", recommendations)

evaluate_algorithm(URM_test, random_recommender, at=10)
