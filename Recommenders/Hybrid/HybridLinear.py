import numpy as np

from Recommenders.BaseRecommender import BaseRecommender


class HybridLinear(BaseRecommender):
    def __init__(self, URM_train, recommenders, verbose=True):
        super().__init__(URM_train, verbose)

        self.recommenders = recommenders

    def fit(self, iALS, ItemKNN, EASE_R, P3alpha, RP3beta, SLIM):
        self.weights = {
            "iALS": iALS,
            "ItemKNN": ItemKNN,
            "EASE_R": EASE_R,
            "P3alpha": P3alpha,
            "RP3beta": RP3beta,
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

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        for rec_id, rec_obj in self.recommenders.items():
            rec_obj.save_model(folder_path, rec_id + "_" + file_name)


