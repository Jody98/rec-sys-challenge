import numpy as np
from scipy.sparse.linalg import svds


class SVDBasedRecommender(object):
    def __init__(self, num_factors=50):
        self.num_factors = num_factors
        self.U = None
        self.S = None
        self.Vt = None

    def fit(self, URM_train):
        U, S, Vt = svds(URM_train, k=self.num_factors)
        self.U = U
        self.S = np.diag(S)
        self.Vt = Vt

    def recommend(self, user_id, at=10):
        user_factors = self.U[user_id, :]
        item_factors = self.Vt
        scores = np.dot(user_factors, item_factors)
        recommended_items = np.argsort(scores)[::-1][:at]

        recommendation = {"user_id": user_id, "item_list": recommended_items}
        return recommendation, recommended_items
