import numpy as np


class RandomRecommender(object):

    def fit(self, URM_train):
        self.n_items = URM_train.shape[1]

    def recommend(self, user_id, at=10):
        recommended_items = np.random.choice(self.n_items, at)
        recommendation = {"user_id": user_id, "item_list": recommended_items}
        return recommendation, recommended_items

    def generate_submission_csv(self, filename, recommendations):
        with open(filename, 'w') as f:
            f.write("user_id,item_list\n")
            for recommendation in recommendations:
                f.write("{},{}\n".format(recommendation["user_id"],
                                         " ".join(str(x) for x in recommendation["item_list"])))
