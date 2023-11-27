from HyperparameterTuning.tune_base_recommender import tune_base_recommender
from Recommenders.MatrixFactorization.IALSRecommender import IALSRecommender

tune_base_recommender(
    recommender_class=IALSRecommender,
    n=100,
    output_folder="iALS",
    save_trained_on_all=True
)