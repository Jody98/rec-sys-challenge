import scipy.sparse as sps

from Evaluation.Evaluator import EvaluatorHoldout
from Recommenders.GraphBased import RP3betaRecommender
from Recommenders.Hybrid.GeneralizedLinearHybridRecommender import GeneralizedLinearHybridRecommender
from Recommenders.KNN import ItemKNNCFRecommender
from Recommenders.KNN.ItemKNNSimilarityHybridRecommender import ItemKNNSimilarityHybridRecommender
from Recommenders.SLIM import SLIMElasticNetRecommender
from challenge.utils.functions import read_data


def __main__():
    cutoff_list = [10]
    data_file_path = '../input_files/data_train.csv'
    users_file_path = '../input_files/data_target_users_test.csv'
    URM_all_dataframe, users_list = read_data(data_file_path, users_file_path)

    URM_train = sps.load_npz("../input_files/URM_train_plus_validation.npz")
    URM_test = sps.load_npz("../input_files/URM_test.npz")

    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    item_recommender = ItemKNNCFRecommender.ItemKNNCFRecommender(URM_train)
    item_recommender.fit(topK=9, shrink=19, similarity='tversky', tversky_alpha=0.036424892090848766,
                         tversky_beta=0.9961018325655608)

    results, _ = evaluator.evaluateRecommender(item_recommender)
    print("ItemKNNCFRecommender MAP: {}".format(results.loc[10]["MAP"]))

    RP3_recommender = RP3betaRecommender.RP3betaRecommender(URM_train)
    RP3_recommender.fit(topK=30, alpha=0.26362900188025656, beta=0.17133265585189086, min_rating=0.2588031389774553,
                        implicit=True, normalize_similarity=True)
    RP3_Wsparse = RP3_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(RP3_recommender)
    print("RP3betaRecommender MAP: {}".format(results.loc[10]["MAP"]))

    SLIM_recommender = SLIMElasticNetRecommender.SLIMElasticNetRecommender(URM_train)
    SLIM_recommender.fit(topK=216, l1_ratio=0.0032465600313226354, alpha=0.002589066655986645, positive_only=True)
    SLIM_Wsparse = SLIM_recommender.W_sparse

    results, _ = evaluator.evaluateRecommender(SLIM_recommender)
    print("SLIM MAP: {}".format(results.loc[10]["MAP"]))

    SLIMRP3 = ItemKNNSimilarityHybridRecommender(URM_train, RP3_Wsparse, SLIM_Wsparse)
    SLIMRP3.fit(topK=48, alpha=0.5304989217384739)

    results, _ = evaluator.evaluateRecommender(SLIMRP3)
    print("SLIMRP3 MAP: {}".format(results.loc[10]["MAP"]))

    recommenders = [item_recommender, item_recommender, item_recommender, RP3_recommender, SLIM_recommender]
    alpha = 0
    beta = 0
    gamma = 0.14621041546759697
    delta = 1.4088635313876665
    epsilon = 1.4578805008713718

    recommender_object = GeneralizedLinearHybridRecommender(URM_train, recommenders=recommenders)
    recommender_object.fit(alpha=alpha, beta=beta, gamma=gamma, delta=delta, epsilon=epsilon)

    results, _ = evaluator.evaluateRecommender(recommender_object)
    print("GeneralizedLinearHybridRecommender")
    print("MAP: {}".format(results.loc[10]["MAP"]))


if __name__ == '__main__':
    __main__()
