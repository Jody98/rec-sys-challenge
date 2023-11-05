import numpy as np


class RandomRecommender(object):
    def __init__(self):
        self.n_items = None

    def fit(self, URM_train):
        self.n_items = URM_train.shape[1]

    def recommend(self, user_id, at=10):
        recommended_items = np.random.choice(self.n_items, at)
        recommendation = {"user_id": user_id, "item_list": recommended_items}
        return recommendation, recommended_items
