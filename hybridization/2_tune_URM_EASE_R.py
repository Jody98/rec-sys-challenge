from HyperparameterTuning.tune_URM import tune_URM
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

tune_URM(recommender_class=EASE_R_Recommender, n=100, recommender_folder="EASE_R")
