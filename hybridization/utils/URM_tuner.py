import numpy as np
from Recommenders.BaseRecommender import (
    BaseRecommender,
)
from data_manager import DatasetLoader, DatasetSplitter, URMGenerator
from evaluation.evaluator import EvaluatorHoldout
from pathlib import Path
from utils.load_best_hyperparameters import load_best_hyperparameters

def URM_tuner(N, recommender_class: BaseRecommender, model_folder: Path):
    dataset_loader = DatasetLoader()
    dataset_splitter = DatasetSplitter(dataset_loader)
    dataset_train, dataset_val = dataset_splitter.load_interactions_train_val()
    URM_generator = URMGenerator(dataset_train, dataset_val)

    best_hyperparameters = load_best_hyperparameters(model_folder)

    best_map = 0
    best_base = 0
    best_views_weight = 0
    best_details_weight = 0
    for i in range(N):
        print(f"\nIter {i}")
        random_base = np.random.uniform(2,100)
        random_details_weight = np.random.uniform(0,100)
        random_views_weight = np.random.uniform(random_details_weight,100)
        print(f"base={random_base}, views_weight={random_views_weight}, details_weight={random_details_weight}")

        URM_train, URM_val = URM_generator.generate_explicit_URM(
            log_base=random_base,
            views_weight=random_views_weight,
            details_weight=random_details_weight
        )

        evaluator = EvaluatorHoldout(URM_val, cutoff_list=[10])
        print("Fitting")
        rec = recommender_class(URM_train)
        rec.fit(**best_hyperparameters)
        print("Evaluation")
        results_df, result_str = evaluator.evaluateRecommender(rec)
        print(result_str)

        result_map = list(results_df["MAP"])[0]
        if result_map > best_map:
            best_map = result_map
            best_base = random_base
            best_views_weight = random_views_weight
            best_details_weight = random_details_weight

        print(f"\nBest MAP so far: {best_map}")
        print(f"base={best_base}, views_weight={best_views_weight}, details_weight={best_details_weight}")

    return best_base, best_views_weight, best_details_weight




