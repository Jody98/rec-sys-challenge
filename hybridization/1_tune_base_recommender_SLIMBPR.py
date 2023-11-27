from HyperparameterTuning.tune_base_recommender import tune_base_recommender
from Recommenders.SLIM.SLIM_BPR_Python import SLIM_BPR_Python

tune_base_recommender(
    recommender_class=SLIM_BPR_Python,
    n=100,
    output_folder="SLIM",
    save_trained_on_all=False
)
