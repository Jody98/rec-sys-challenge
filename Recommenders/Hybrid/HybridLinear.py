import numpy as np

from Recommenders.BaseRecommender import BaseRecommender


class HybridLinear(BaseRecommender):
    def __init__(self, URM_train, recommenders, verbose=True):
        super().__init__(URM_train, verbose)

        self.recommenders = recommenders

    def fit(self, KNN, RP3beta, iALS, EASE_R, SLIM):
        self.weights = {
            "KNN": KNN,
            "RP3beta": RP3beta,
            "iALS": iALS,
            "EASE_R": EASE_R,
            "SLIM": SLIM
        }

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        item_weights = {}
        for rec_id, rec_obj in self.recommenders.items():
            rec_item_weights = rec_obj._compute_item_score(user_id_array)
            mean = np.mean(rec_item_weights)
            std = np.std(rec_item_weights)
            item_weights[rec_id] = (rec_item_weights - mean) / std

        result = 0
        for rec_id in self.recommenders.keys():
            result += item_weights[rec_id] * self.weights[rec_id]

        return result
