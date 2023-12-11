#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/04/18

@author: Maurizio Ferrari Dacrema
"""

from Recommenders.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender
from Recommenders.Recommender_utils import check_matrix, similarityMatrixTopK


class ItemKNNSimilarityTripleHybridRecommender(BaseItemSimilarityMatrixRecommender):
    """ ItemKNNSimilarityHybridRecommender
    Hybrid of two similarities S = S1*alpha + S2*(1-alpha)

    """

    RECOMMENDER_NAME = "ItemKNNSimilarityTripleHybridRecommender"

    def __init__(self, URM_train, Similarity_1, Similarity_2, Similarity_3, verbose=True):
        super(ItemKNNSimilarityTripleHybridRecommender, self).__init__(URM_train, verbose=verbose)

        if Similarity_1.shape != Similarity_2.shape:
            raise ValueError(
                "ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S2 is {}".format(
                    Similarity_1.shape, Similarity_2.shape
                ))

        if Similarity_1.shape != Similarity_3.shape:
            raise ValueError(
                "ItemKNNSimilarityHybridRecommender: similarities have different size, S1 is {}, S3 is {}".format(
                    Similarity_1.shape, Similarity_3.shape
                ))

        if Similarity_2.shape != Similarity_3.shape:
            raise ValueError(
                "ItemKNNSimilarityHybridRecommender: similarities have different size, S2 is {}, S3 is {}".format(
                    Similarity_2.shape, Similarity_3.shape
                ))

        # CSR is faster during evaluation
        self.Similarity_1 = check_matrix(Similarity_1.copy(), 'csr')
        self.Similarity_2 = check_matrix(Similarity_2.copy(), 'csr')
        self.Similarity_3 = check_matrix(Similarity_3.copy(), 'csr')

    def fit(self, topK=100, alpha=0.25, beta=0.25):
        self.topK = topK
        self.alpha = alpha
        self.beta = beta

        W_sparse = self.Similarity_1 * self.alpha + self.Similarity_2 * self.beta + self.Similarity_3 * (
                    1 - self.alpha - self.beta)

        self.W_sparse = similarityMatrixTopK(W_sparse, k=self.topK)
        self.W_sparse = check_matrix(self.W_sparse, format='csr')
