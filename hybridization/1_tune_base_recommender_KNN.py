from HyperparameterTuning.tune_base_recommender import tune_base_recommender
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

tune_base_recommender(
    recommender_class=ItemKNNCFRecommender,
    n=100,
    output_folder="KNN",
    save_trained_on_all=False
)