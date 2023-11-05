import numpy as np


class TopPopRecommender(object):
    def __init__(self):
        self.popular_items = None

    def fit(self, URM_train):
        item_popularity = np.ediff1d(URM_train.tocsc().indptr)
        self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.flip(self.popular_items, axis=0)

    def recommend(self, user_id, at=10):
        recommended_items = self.popular_items[0:at]
        recommendation = {"user_id": user_id, "item_list": recommended_items}
        return recommendation, recommended_items


class TopPopRecommenderRemoved(object):

    def __init__(self):
        self.popular_items = None
        self.n_items = None

    def fit(self, URM_train):
        self.n_items = URM_train.shape[1]
        item_popularity = (URM_train > 0).sum(axis=0)
        item_popularity = np.array(item_popularity).squeeze()
        self.popular_items = np.argsort(item_popularity)[::-1]

    def recommend(self, user_id, at=10):
        recommended_items = self.popular_items[0:at]
        recommended_items = np.delete(recommended_items, np.where(recommended_items == user_id))
        recommendation = {"user_id": user_id, "item_list": recommended_items}
        return recommendation, recommended_items
