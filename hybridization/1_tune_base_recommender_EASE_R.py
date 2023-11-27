from HyperparameterTuning.tune_base_recommender import tune_base_recommender
from Recommenders.EASE_R.EASE_R_Recommender import EASE_R_Recommender

tune_base_recommender(
    recommender_class=EASE_R_Recommender,
    n=100,
    output_folder="EASE_R",
    save_trained_on_all=False
)