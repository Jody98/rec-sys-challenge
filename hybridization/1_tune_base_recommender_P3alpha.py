from HyperparameterTuning.tune_base_recommender import tune_base_recommender
from Recommenders.GraphBased.RP3betaRecommender import RP3betaRecommender

tune_base_recommender(
    recommender_class=RP3betaRecommender,
    n=100,
    output_folder="RP3beta",
    save_trained_on_all=False
)