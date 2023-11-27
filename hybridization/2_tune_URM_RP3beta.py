from hyperparameter_tuning.tune_URM import tune_URM
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

tune_URM(recommender_class=RP3betaRecommender, n=100, recommender_folder="RP3beta")
