import os
from pathlib import Path
from datetime import datetime

import pandas as pd

from Recommenders.BaseRecommender import BaseRecommender


def create_submission(
    rec: BaseRecommender,
    data_path: Path = Path("data"),
    output_path: Path = Path("submissions"),
):
    """Create a submission for a recommender.

    Args:
        rec (BaseRecommender): The recommender.
        data_path (Path, optional): The path where to find the target users.. Defaults to Path("data").
        output_path (Path, optional): The path where to save the submission. Defaults to Path("submissions").

    Raises:
        FileNotFoundError: If data_path or output_path/data_target_users_test.csv does not exists.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The path {str(data_path)} does not exists.")
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"The path {str(output_path)} does not exists.")

    target_user_file = data_path / "data_target_users_test.csv"
    if not os.path.isfile(data_path / "data_target_users_test.csv"):
        raise FileNotFoundError(f"The path {str(target_user_file)} does not exists.")

    target_users = pd.read_csv(target_user_file, dtype={0: int})["user_id"].values

    print("Generating recommendations...")
    recommendations = rec.recommend(target_users, cutoff=10)

    now: str = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_file_name = f"{rec.RECOMMENDER_NAME}-{now}.csv"
    output_path = os.path.join(output_path, output_file_name)

    print(f"Writing to {output_file_name}...")
    with open(output_path, "w") as f:
        f.write("user_id,item_list\n")
        for i in range(len(target_users)):
            f.write(
                str(target_users[i])
                + ","
                + " ".join(str(x) for x in recommendations[i])
                + "\n"
            )
