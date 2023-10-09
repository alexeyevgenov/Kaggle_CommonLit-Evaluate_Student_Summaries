import pandas as pd

import optuna_search_ridge
from config import CONFIG
from utils.data_processing import group_folds_in_a_single_df


def ridge_study(data: pd.DataFrame) -> None:
    for target in CONFIG.data.targets:
        study = optuna_search_ridge.Study(data, target)
        study.search(n_trials=CONFIG.n_trials)
        study.train_best_model()


if __name__ == "__main__":
    data = group_folds_in_a_single_df(CONFIG.storage, CONFIG.num_folds)
    targets = data[CONFIG.data.targets]
    emb_columns = [el for el in data.columns if "emb_" in el]
    fold_nums = data["fold"]
    emb_features = data[emb_columns]

    data = pd.concat([emb_features, fold_nums, targets], axis=1)

    # study
    ridge_study(data)
