import warnings

import pandas as pd

import optuna_search_lgbm
import optuna_search_svr
from config import CONFIG
from feature_generation.data_processing_unit import group_folds_in_a_single_df, remove_highly_collinear_variables

MODEL_TYPE = "SVR"  # ["GBM", "SVR", "RIDGE"]


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
    all_folds_df = remove_highly_collinear_variables(group_folds_in_a_single_df(CONFIG.storage, CONFIG.num_folds),
                                                     collinearity_threshold=0.95)

    # study
    if MODEL_TYPE == "GBM":
        gbm_study(all_folds_df)
    elif MODEL_TYPE == "SVR":
        svr_study(all_folds_df)
    elif MODEL_TYPE == "RIDGE":
        ridge_study(all_folds_df)
    else:
        raise KeyError("Chosen type of model is incorrect")
