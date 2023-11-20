import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from challenge.experiments.cython_SLIM_MSE import train_multiple_epochs
from utils.functions import read_data


def __main__():
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    learning_rate = 1e-3
    epochs = 100
    regularization = 0.0

    item_item_S, loss, samples_per_second = train_multiple_epochs(URM_train=URM_train,
                                                                  learning_rate_input=learning_rate,
                                                                  regularization_2_input=regularization,
                                                                  n_epochs=epochs)

    print("Loss: {}, samples_per_second: {}".format(loss, samples_per_second))


if __name__ == '__main__':
    __main__()
