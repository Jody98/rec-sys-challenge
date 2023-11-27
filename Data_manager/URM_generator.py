from typing import Tuple

import pandas as pd
import numpy as np
import scipy.sparse as sps


class URMGenerator(object):
    def __init__(
        self, interactions_train: pd.DataFrame, interactions_val: pd.DataFrame
    ) -> None:
        self.interactions_train = interactions_train.copy()
        self.interactions_val = interactions_val.copy()

        # Find n_users and n_items
        users_id = np.unique(
            np.concatenate(
                (self.interactions_train["user_id"], self.interactions_val["user_id"])
            )
        )
        items_id = np.unique(
            np.concatenate(
                (self.interactions_train["item_id"], self.interactions_val["item_id"])
            )
        )
        self.n_users = users_id.shape[0]
        self.n_items = items_id.shape[0]

    def _generate_explicit_split_URM(
        self, dataset: pd.DataFrame, log_base=2, views_weight=1, details_weight=1
    ) -> sps.coo_matrix:
        dataset["combined_ratings"] = (np.log(
            views_weight * dataset["views_count"]
            + details_weight * dataset["details_count"]
            + 1
        ) / np.log(log_base))

        URM = sps.coo_matrix(
            (
                dataset["combined_ratings"].values.astype("float64"),
                (dataset["user_id"].values, dataset["item_id"].values),
            ),
            shape=(self.n_users, self.n_items),
        )
        return URM

    def generate_explicit_URM(
        self, log_base=2, views_weight=1, details_weight=1
    ) -> Tuple[sps.coo_matrix, sps.coo_matrix]:
        print("Generating explicit URM...")

        URM_train = self._generate_explicit_split_URM(
            self.interactions_train, log_base, views_weight, details_weight
        )
        URM_val = self._generate_explicit_split_URM(
            self.interactions_val, log_base, views_weight, details_weight
        )

        return URM_train, URM_val

    def _geneate_impicit_split_URM(self, dataset: pd.DataFrame) -> sps.coo_matrix:
        dataset["interacted"] = 1

        URM = sps.coo_matrix(
            (
                dataset["interacted"].values,
                (dataset["user_id"].values, dataset["item_id"].values),
            ),
            shape=(self.n_users, self.n_items),
        )
        return URM

    def generate_implicit_URM(self) -> Tuple[sps.coo_matrix, sps.coo_matrix]:
        print("Generating implicit URM...")

        URM_train = self._geneate_impicit_split_URM(self.interactions_train)
        URM_val = self._geneate_impicit_split_URM(self.interactions_val)

        return URM_train, URM_val
