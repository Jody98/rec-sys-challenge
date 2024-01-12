import pickle
import time
from itertools import combinations
from warnings import simplefilter

import numpy as np
import pandas as pd
import scipy.sparse as sps
from Recommenders.ScoresHybridRecommender import ScoresHybridRecommender
from category_encoders import TargetEncoder
from sklearn.discriminant_analysis import StandardScaler
from tqdm import tqdm
from xgboost import XGBRanker, plot_importance

from Recommenders.BaseRecommender import BaseRecommender

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class XGboostRecommender(BaseRecommender):
    """XGboost recommender"""

    RECOMMENDER_NAME = "XGboostRecommender"

    def __init__(self, URM_train, URM_val, load_model_path=None, categorical_method=None):
        super(XGboostRecommender, self).__init__(URM_train)
        self.n_users, self.n_items = self.URM_train.shape
        self.URM_train = URM_train
        self.URM_val = URM_val
        self.categorical_method = categorical_method

        # ------- Recs -----------
        with open('best_models_info.pickle', 'rb') as f:
            self.best_models_info = pickle.load(f)

        self.models_to_load = [
            'ItemKNNCFRecommenderCrossValNDCG',
            'UserKNNCFRecommenderCrossValNDCG',
            'P3alphaRecommenderCrossValNDCG',
            'RP3betaRecommenderCrossValNDCG',
            'SLIMElasticNetRecommenderCrossValNDCG',
            'TopPop',
            'IALSRecommender',
            'MultVAERecommender',
            'SLIMElasticNetRecommenderCrossValRecall40',
            'RP3betaRecommenderCrossValRecall40',
        ]

        self.models_to_load_info = {k: v for k, v in self.best_models_info.items() if k in self.models_to_load}
        self.recs = self.train_models(URM_train, load_model_path=load_model_path)
        self.cutoff_df_map = {}
        # ------------------
        self.labels = self._compute_labels(URM_val)
        self.features = None

    def train_models(self, URM, load_model_path=None):
        recs = {}
        for rec_name, rec_info in tqdm(self.models_to_load_info.items()):
            rec_instance = rec_info['instance'](URM)
            if (load_model_path is None):
                rec_instance.fit(**rec_info['hs'])
            else:
                rec_instance.load_model(load_model_path, rec_name)
            recs[rec_name] = rec_instance

        hybrid_name = 'ScoresHybridRecommenderCrossValNDCG'
        hybrid_info = self.best_models_info[hybrid_name]
        base_recs = [recs[base_rec] for base_rec in hybrid_info['base_recs'] if base_rec in recs]
        hybrid_rec = ScoresHybridRecommender(URM, base_recs[0], base_recs[1])
        hybrid_rec.fit(**hybrid_info['hs'])
        recs[hybrid_name] = hybrid_rec

        hybrid_name = 'ScoresHybridRecommenderCrossValRecall40'
        hybrid_info = self.best_models_info[hybrid_name]
        base_recs = [recs[base_rec] for base_rec in hybrid_info['base_recs'] if base_rec in recs]
        hybrid_rec = ScoresHybridRecommender(URM, base_recs[0], base_recs[1])
        hybrid_rec.fit(**hybrid_info['hs'])
        recs[hybrid_name] = hybrid_rec

        return recs

    def prepare_to_submit(self, URM, cutoff=35, load_model_path=None):
        final_recs = self.train_models(URM, load_model_path=load_model_path)
        self.features = self._compute_features(URM, final_recs, cutoff=cutoff)

    def fit(self,
            cutoff=35,
            n_estimators=50,
            learning_rate=1e-1,
            reg_alpha=1e-1,
            reg_lambda=1e-1,
            max_depth=5,
            max_leaves=0,
            grow_policy="depthwise",
            objective="rank:pairwise",
            booster="gbtree",
            tree_method="hist",
            random_state=None,
            verbosity=0):

        if (cutoff in self.cutoff_df_map):
            self.features = self.cutoff_df_map[cutoff]
        else:
            self.features = self._compute_features(self.URM_train, self.recs, cutoff=cutoff)
            self.cutoff_df_map[cutoff] = self.features

        self.model = XGBRanker(objective=objective,
                               n_estimators=int(n_estimators),
                               random_state=random_state,
                               learning_rate=learning_rate,
                               reg_alpha=reg_alpha,
                               reg_lambda=reg_lambda,
                               max_depth=int(max_depth),
                               max_leaves=int(max_leaves),
                               grow_policy=grow_policy,
                               verbosity=verbosity,
                               booster=booster,
                               enable_categorical=self.categorical_method is None,
                               tree_method=tree_method,
                               )

        X_train = self.features.drop(columns=["Label"])
        y_train = self.features["Label"]
        groups = self.features.groupby("UserID").size().values

        if (self.categorical_method != None):
            categorical_columns = X_train.select_dtypes(include=['category']).columns
            X_train.drop(columns=categorical_columns, inplace=True)

        self.model.fit(X_train,
                       y_train,
                       group=groups,
                       verbose=True)

    def _compute_labels(self, URM):
        URM_coo = sps.coo_matrix(URM)
        df = pd.DataFrame({"UserID": URM_coo.row, "ItemID": URM_coo.col})
        df['UserID'] = df['UserID'].astype('category')
        df['ItemID'] = df['ItemID'].astype('category')

        return df

    def _compute_features(self, URM, recs, cutoff, load=False):
        if (not load):
            start_time = time.time()
            df = pd.DataFrame(index=range(0, self.n_users), columns=["ItemID"])
            df.index.name = 'UserID'
            df.reset_index(inplace=True)
            df.rename(columns={"index": "UserID"}, inplace=True)

            generators_name = ['ScoresHybridRecommenderCrossValRecall40']
            generators = [rec_instance for rec_name, rec_instance in recs.items() if rec_name in generators_name]
            df = self.generate_candidate_from_multiple(df, generators, cutoff=cutoff)

            features_rec_instances = [rec_instance for rec_name, rec_instance in recs.items() if
                                      "Recall" not in rec_name]
            df = self.generate_recs_scores(df, features_rec_instances)
            k_values = [3, 10]
            df = self.generate_is_top_k_feature(df, features_rec_instances, k_values=k_values)
            df = self.generate_count_agreement_on_k_feature(df, features_rec_instances, k_values=k_values)
            df = self.generate_recs_stats(df, features_rec_instances)
            df = self.normalize_recs_scores(df, features_rec_instances)
            df = self.generate_dissimilarity_features(df, features_rec_instances)

            n_groups = 20
            df = self.generate_user_profile(df, URM, n_groups=n_groups)
            df = self.generate_item_profile(df, URM, n_groups=n_groups)

            df = self.merge_feature_and_label(df, self.labels)

            if (self.categorical_method == 'target'):
                df = self.encode_categoricals_target(df)
            elif (self.categorical_method == 'svd'):
                df = self.encode_categoricals_svd(df, latent_dims=70)

            df[df.select_dtypes(include='float64').columns] = df.select_dtypes(include='float64').astype(np.float32)
            print(f'Time to compute features: {((time.time() - start_time) / 60):.1f}m')
        else:
            df = pd.read_pickle(f'tmp_df/df_xgboost_{self.categorical_method}.pkl')

        return df

    def generate_candidate_from_multiple(self, df: pd.DataFrame, generators, cutoff=35):
        df = df.copy()
        df.set_index("UserID", inplace=True)
        for user_id in tqdm(range(self.n_users), desc='Generating candidates'):
            recommendations = set()
            for generator in generators:
                recommendations.update(generator.recommend(user_id, cutoff=cutoff))
            df.loc[user_id, "ItemID"] = list(recommendations)
        df = df.explode("ItemID").astype(int)
        df.reset_index(inplace=True)
        return df

    def generate_recs_scores(self, df: pd.DataFrame, other_recs):
        df = df.copy()
        df.set_index("UserID", inplace=True)
        for user_id in tqdm(range(self.n_users), desc='Generating recs scores'):
            for rec_instance in other_recs:
                item_list = df.loc[user_id, "ItemID"].values.tolist()
                all_item_scores = rec_instance._compute_item_score([user_id], items_to_compute=item_list)
                df.loc[user_id, rec_instance.RECOMMENDER_NAME] = all_item_scores[0, item_list]
        df.reset_index(inplace=True)
        return df

    def generate_recs_stats(self, df: pd.DataFrame, other_recs):
        def iqr(x):
            return x.quantile(0.75) - x.quantile(0.25)

        df = df.copy()
        to_compute_stats = [rec.RECOMMENDER_NAME for rec in other_recs]
        print('Computing aggregate stats...')
        stats = df.groupby('UserID')[to_compute_stats].agg([
            'mean',
            'std',
            'median',
            iqr,
            'skew',
        ])
        stats.columns = ['_user_'.join(col).strip() for col in stats.columns.values]
        df = df.merge(stats, left_on='UserID', right_index=True)
        return df

    def normalize_recs_scores(self, df: pd.DataFrame, other_recs):
        df = df.copy()
        to_normalize = [rec.RECOMMENDER_NAME for rec in other_recs]

        def standardize_group(group):
            return StandardScaler().fit_transform(group.values.reshape(-1, 1)).flatten()

        df[to_normalize] = df.groupby('UserID')[to_normalize].transform(standardize_group)
        return df

    def generate_user_profile(self, df: pd.DataFrame, URM, n_groups=20):
        df = df.copy()
        user_popularity = np.ediff1d(sps.csr_matrix(URM).indptr)
        df['user_profile_len'] = user_popularity[df["UserID"].values.astype(int)]

        df.set_index('UserID', inplace=True)
        block_size = int(len(user_popularity) * 0.05)
        sorted_users = np.argsort(user_popularity)
        users_recommended = df.index.unique().values.astype(int)

        for group_id in tqdm(range(0, n_groups), desc='Generating user profile'):
            start_pos = group_id * block_size
            end_pos = min((group_id + 1) * block_size, len(user_popularity)) if group_id != n_groups - 1 else len(
                user_popularity)

            users_in_group = sorted_users[start_pos:end_pos]
            users_in_group_p_len = user_popularity[users_in_group]

            idx = [user for user in users_in_group if user in users_recommended]
            df.loc[idx, 'user_popularity_group_id'] = int(group_id)
            df.loc[idx, 'user_popularity_group_p_len_mean'] = np.mean(users_in_group_p_len)
            df.loc[idx, 'user_popularity_group_p_len_median'] = np.median(users_in_group_p_len)

        df['user_popularity_group_id'] = df['user_popularity_group_id'].astype('category')
        df.reset_index(inplace=True)
        return df

    def generate_item_profile(self, df: pd.DataFrame, URM, n_groups=20):
        df = df.copy()
        item_popularity = np.ediff1d(sps.csc_matrix(URM).indptr)
        df['item_profile_len'] = item_popularity[df["ItemID"].values.astype(int)]

        df.set_index('ItemID', inplace=True)
        block_size = int(len(item_popularity) * 0.05)
        sorted_items = np.argsort(item_popularity)
        items_recommended = df.index.unique().values.astype(int)

        for group_id in tqdm(range(0, n_groups), desc='Generating item profile'):
            start_pos = group_id * block_size
            end_pos = min((group_id + 1) * block_size, len(item_popularity)) if group_id != n_groups - 1 else len(
                item_popularity)

            items_in_group = sorted_items[start_pos:end_pos]
            items_in_group_p_len = item_popularity[items_in_group]

            idx = [item for item in items_in_group if item in items_recommended]
            df.loc[idx, 'item_popularity_group_id'] = int(group_id)
            df.loc[idx, 'item_popularity_group_p_len_mean'] = np.mean(items_in_group_p_len)
            df.loc[idx, 'item_popularity_group_p_len_median'] = np.median(items_in_group_p_len)

        df['item_popularity_group_id'] = df['item_popularity_group_id'].astype('category')
        df.reset_index(inplace=True)

        return df

    def generate_is_top_k_feature(self, df: pd.DataFrame, other_recs, k_values):
        df = df.copy()
        for rec_instance in tqdm(other_recs, desc='Generating is top k feature'):
            for k in k_values:
                df[f'is_in_top_{k}_{rec_instance.RECOMMENDER_NAME}'] = df.groupby('UserID')[
                                                                           rec_instance.RECOMMENDER_NAME].rank(
                    ascending=False, method='first') <= k
        return df

    def generate_count_agreement_on_k_feature(self, df: pd.DataFrame, other_recs, k_values):
        df = df.copy()
        for k in tqdm(k_values, desc='Generating count agreement on k feature'):
            columns = [f'is_in_top_{k}_{rec_instance.RECOMMENDER_NAME}' for rec_instance in other_recs]
            df[f'rec_agree_on_{k}'] = df[columns].sum(axis=1).astype(np.int32)
        return df

    def generate_dissimilarity_features(self, df: pd.DataFrame, other_recs):
        df = df.copy()
        model_pairs = list(combinations([f'{rec_instance.RECOMMENDER_NAME}' for rec_instance in other_recs], 2))
        for pair in tqdm(model_pairs, desc='Generating user based dissimilarity features'):
            pair1, pair2 = pair
            df[f'{pair1}_{pair2}_abs_diff'] = (df[pair1] - df[pair2]).abs()
        return df

    def encode_categoricals_svd(self, df: pd.DataFrame, latent_dims=100):
        df = df.copy()
        categorical_columns = df.select_dtypes(include=['category']).columns
        print(f'SVD encoding categorical columns: {categorical_columns.tolist()}...')

        X_seq = df[categorical_columns].apply(lambda x: " ".join(list([str(y) + str(i) for i, y in enumerate(x)])),
                                              axis=1)

        svd_feats = ['svd_' + str(l) for l in range(latent_dims)]
        vectorizer = TfidfVectorizer()

        dim_reduction = TruncatedSVD(n_components=latent_dims, random_state=0)
        df[svd_feats] = dim_reduction.fit_transform(vectorizer.fit_transform(X_seq))

        cumulative_explained_variance = dim_reduction.explained_variance_ratio_.cumsum()
        explained_variance = cumulative_explained_variance[-1]

        print(f'Variance explained by {latent_dims} latent dimensions: {explained_variance:.2%}')
        return df

    def encode_categoricals_target(self, df: pd.DataFrame, smoothing=0.20):
        df_enc = df.copy()
        categorical_columns = df_enc.select_dtypes(include=['category']).columns
        encoded_columns = [f'{col}_enc' for col in categorical_columns]
        print(f'Target encoding categorical columns: {categorical_columns.tolist()}...')

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        X = df_enc.drop(columns=['Label'])
        y = df_enc['Label']
        groups = X['UserID'].astype(int)
        df_enc[encoded_columns] = np.nan
        for train_idx, val_idx in tqdm(sgkf.split(X, y, groups=groups), desc='Encoding categorical columns'):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val, _ = X.iloc[val_idx], y.iloc[val_idx]
            encoder = TargetEncoder(cols=categorical_columns, smoothing=smoothing, target_type='binary')
            encoder.fit(X_train, y_train)
            encoded_values = encoder.transform(X_val)
            df_enc.loc[encoded_values.index, encoded_columns] = encoded_values[categorical_columns].values

        ce_target_encoder = TargetEncoder(cols=categorical_columns, smoothing=smoothing, target_type='binary')
        ce_target_encoder.fit(df_enc, df_enc['Label'])
        self.ce_target_encoder = ce_target_encoder

        return df_enc

    def merge_feature_and_label(self, features, labels):
        df = pd.merge(features, labels, on=['UserID', 'ItemID'], how='left', indicator='Exist')
        df["Label"] = df["Exist"].apply(lambda x: 1 if x == 'both' else 0)
        df.drop(columns=['Exist'], inplace=True)

        df = df.sort_values("UserID").reset_index()
        df.drop(columns=['index'], inplace=True)

        df["UserID"] = df["UserID"].astype("category")
        df["ItemID"] = df["ItemID"].astype("category")

        return df

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        user_mask = self.features["UserID"].isin(user_id_array)
        X_to_predict_user = self.features[user_mask].copy()
        X_to_predict_user.drop(columns=["Label"], inplace=True)

        user_ids = X_to_predict_user['UserID'].astype(int).values
        item_ids = X_to_predict_user['ItemID'].astype(int).values
        item_scores = np.full((self.n_users, self.n_items), -np.inf, dtype=np.float32)

        if (self.categorical_method is not None):
            categorical_columns = X_to_predict_user.select_dtypes(include=['category']).columns
            X_to_predict_user.drop(columns=categorical_columns, inplace=True)

        scores = self.model.predict(X_to_predict_user)
        item_scores[user_ids, item_ids] = scores

        return item_scores[user_id_array]

    def plot_importance(self, max_num_features=30):
        plot_importance(self.model, importance_type='weight', title='Weight (Frequence)',
                        max_num_features=max_num_features)
