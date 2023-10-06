import warnings

import argparse
import pandas as pd

import optuna_search_lgbm
import optuna_search_svr
import optuna_search_ridge
from config import CONFIG
from utils.data_processing import group_folds_in_a_single_df, remove_highly_collinear_variables


def gbm_study(data: pd.DataFrame) -> None:
    warnings.filterwarnings("ignore")
    for target in CONFIG.data.targets:
        study = optuna_search_lgbm.Study(data, target)
        study.search(n_trials=CONFIG.n_trials)
        study.train_best_model()


def svr_study(data: pd.DataFrame) -> None:
    for target in CONFIG.data.targets:
        study = optuna_search_svr.Study(data, target)
        study.search(n_trials=CONFIG.n_trials)
        study.train_best_model()


def ridge_study(data: pd.DataFrame) -> None:
    for target in CONFIG.data.targets:
        study = optuna_search_ridge.Study(data, target)
        study.search(n_trials=CONFIG.n_trials)
        study.train_best_model()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str,
                        help="model selection ['GBM', 'SVR', 'RIDGE']")
    arguments = parser.parse_args()

    data = group_folds_in_a_single_df(CONFIG.storage, CONFIG.num_folds)
    features = data.drop(columns=CONFIG.data.targets)
    targets = data[CONFIG.data.targets]
    features = remove_highly_collinear_variables(features, collinearity_threshold=CONFIG.feat_coll_thresh)
    data = pd.concat([features, targets], axis=1)

    # study
    if arguments.model == "GBM":
        gbm_study(data)
    elif arguments.model == "SVR":
        svr_study(data)
    elif arguments.model == "RIDGE":
        ridge_study(data)
    else:
        raise KeyError("Chosen type of model is incorrect")
