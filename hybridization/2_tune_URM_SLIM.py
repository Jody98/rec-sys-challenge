from hyperparameter_tuning.tune_URM import tune_URM
from Recommenders.SLIM.SLIMElasticNetRecommender import SLIMElasticNetRecommender

tune_URM(recommender_class=SLIMElasticNetRecommender, n=100, recommender_folder="SLIM")
