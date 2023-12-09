import scipy.sparse as sps

from Data_manager.split_functions.split_train_validation_random_holdout import \
    split_train_in_two_percentage_global_sample
from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.MatrixFactorization.Cython import MatrixFactorization_Cython
from challenge.utils.functions import read_data, generate_submission_csv, evaluate_algorithm


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_all = sps.coo_matrix(
        (URM_all_dataframe['Data'].values, (URM_all_dataframe['UserID'].values, URM_all_dataframe['ItemID'].values)))
    URM_all = URM_all.tocsr()

    URM_train, URM_test = split_train_in_two_percentage_global_sample(URM_all, train_percentage=0.80)

    recommender = MatrixFactorization_Cython.MatrixFactorization_BPR_Cython(URM_train)
    recommender.fit(batch_size=2000,
                    num_factors=100,
                    learning_rate=0.01,
                    use_bias=True,
                    use_embeddings=True,
                    sgd_mode='sgd',
                    negative_interactions_quota=0.0,
                    WARP_neg_item_attempts=10,
                    init_mean=0.0, init_std_dev=0.1,
                    user_reg=0.0, item_reg=0.0, bias_reg=0.0, positive_reg=0.0, negative_reg=0.0)

    recommended_items = recommender.recommend(users_list, cutoff=10)
    recommendations = []
    for i in zip(users_list, recommended_items):
        recommendation = {"user_id": i[0], "item_list": i[1]}
        recommendations.append(recommendation)

    generate_submission_csv("../output_files/MF_BPR_Submission.csv", recommendations)

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)
    results, _ = evaluator.evaluateRecommender(recommender)

    print("MAP: {}".format(results.loc[10]["MAP"]))

    evaluate_algorithm(URM_test, recommender, cutoff_list[0])


if __name__ == '__main__':
    __main__()
