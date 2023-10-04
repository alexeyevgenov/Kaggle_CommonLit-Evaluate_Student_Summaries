import warnings

import argparse
import pandas as pd

import optuna_search_lgbm
import optuna_search_svr
import optuna_search_ridge
from config import CONFIG
from feature_generation.data_processing_unit import group_folds_in_a_single_df, remove_highly_collinear_variables


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
                        help="model selection")
    arguments = parser.parse_args()

    all_folds_df = remove_highly_collinear_variables(group_folds_in_a_single_df(CONFIG.storage, CONFIG.num_folds),
                                                     collinearity_threshold=0.95)

    # study
    if arguments.model == "GBM":
        gbm_study(all_folds_df)
    elif arguments.model == "SVR":
        svr_study(all_folds_df)
    elif arguments.model == "RIDGE":
        ridge_study(all_folds_df)
    else:
        raise KeyError("Chosen type of model is incorrect")
