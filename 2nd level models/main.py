import pandas as pd
from feature_generation.data_processing_unit import group_folds_in_a_single_df
from config import CONFIG
import warnings
import optuna_search_lgbm


def gbm_study(data: pd.DataFrame) -> None:
    warnings.filterwarnings("ignore")
    for target in CONFIG.data.targets:
        study = optuna_search_lgbm.Study(data, target)
        study.search(n_trials=CONFIG.n_trials)
        study.train_best_model()


if __name__ == "__main__":
    all_folds_df = group_folds_in_a_single_df()
    # gbm study
    gbm_study(all_folds_df)
