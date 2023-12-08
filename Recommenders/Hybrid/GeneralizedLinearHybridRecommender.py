# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from Recommenders.BaseRecommender import BaseRecommender


class GeneralizedLinearHybridRecommender(BaseRecommender):
    """
    This recommender merges N recommenders by weighting their ratings
    """

    RECOMMENDER_NAME = "GeneralizedLinearHybridRecommender"

    def __init__(self, URM_train, recommenders: list, verbose=True):

        self.RECOMMENDER_NAME = 'HybridRecommender'

        super(GeneralizedLinearHybridRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommenders = recommenders

    def fit(self, alpha=None, beta=None, gamma=None, delta=None, epsilon=None, alphas=None):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.alphas = [alpha, beta, gamma, delta, epsilon]

    def save_model(self, folder_path, file_name=None):
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        result = self.alphas[0] * self.recommenders[0]._compute_item_score(user_id_array, items_to_compute)
        for index in range(1, len(self.alphas)):
            result = result + self.alphas[index] * self.recommenders[index]._compute_item_score(user_id_array,
                                                                                                items_to_compute)
        return result
