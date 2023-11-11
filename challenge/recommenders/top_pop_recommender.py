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
        self.URM_train = URM_train

        item_popularity = np.ediff1d(URM_train.tocsc().indptr)
        self.popular_items = np.argsort(item_popularity)
        self.popular_items = np.flip(self.popular_items, axis=0)

    def recommend(self, user_id, remove_seen, at=10):
        if remove_seen:
            seen_items = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]
            unseen_items_mask = np.in1d(self.popular_items, seen_items, assume_unique=True, invert=True)
            unseen_items = self.popular_items[unseen_items_mask]
            recommended_items = unseen_items[0:at]
        else:
            recommended_items = self.popular_items[0:at]

        recommendation = {"user_id": user_id, "item_list": recommended_items}
        return recommendation, recommended_items
