from HyperparameterTuning.tune_URM import tune_URM
from Recommenders.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

tune_URM(recommender_class=ItemKNNCFRecommender, n=100, recommender_folder="KNN")
