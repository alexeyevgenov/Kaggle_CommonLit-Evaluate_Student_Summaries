import os
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error
from config import CONFIG
import lightgbm as lgb
from utils.data_processing import normalize_data, split_data_on_train_test
import pickle as pkl
import pandas as pd


class Study:
    def __init__(self, data, target):
        self.target = target
        self.data = data
        self.models = {}
        self.rmses = []
        self.path_for_models_storage = f"{CONFIG.models_dir}/{CONFIG.version}"

        self.df_by_folds = self.prepare_data_for_study()

        storage_for_db = os.path.join(CONFIG.db_dir, CONFIG.version)
        os.makedirs(storage_for_db, exist_ok=True)
        storage = optuna.storages.RDBStorage(url=f"sqlite:///{os.path.join(storage_for_db, f'lgbm_{self.target}.db')}")

        self.study = optuna.create_study(direction='minimize',
                                         study_name=CONFIG.version,
                                         storage=storage,
                                         load_if_exists=True)

    def prepare_data_for_study(self) -> dict:
        df_by_folds = {}
        for fold in range(CONFIG.num_folds):
            X_train, X_eval, y_train, y_eval = split_data_on_train_test(self.data, fold, self.target)
            X_train, X_eval, scaler = normalize_data(X_train, X_eval)
            df_by_folds[fold] = X_train, X_eval, y_train, y_eval

            # store scaler model
            if self.target == CONFIG.data.targets[0]:
                pkl.dump(
                    scaler, open(f"{self.path_for_models_storage}/scaler_{fold}.pkl", "wb"))

        return df_by_folds

    def _objective(self, trial):
        CONFIG.lgbm.n_estimators = trial.suggest_int('n_estimators', 100, 500)
        CONFIG.lgbm.learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
        CONFIG.lgbm.max_depth = trial.suggest_int('max_depth', 2, 5)
        CONFIG.lgbm.num_leaves = trial.suggest_int('num_leaves', 2, 5)
        CONFIG.lgbm.reg_alpha = trial.suggest_float('reg_alpha', 0.0, 10.0)
        CONFIG.lgbm.reg_lambda = trial.suggest_float('reg_lambda', 0.0, 10.0)
        CONFIG.lgbm.min_child_samples = trial.suggest_int('min_child_samples', 0.0, 10.0)

        rmses = []
        for fold in range(CONFIG.num_folds):
            X_train, X_eval, y_train, y_eval = self.df_by_folds[fold]
            model = lgb.LGBMRegressor(**CONFIG.lgbm)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_eval)

            rmse = mean_squared_error(y_eval, y_pred, squared=False)
            rmses.append(rmse)

        print(f"RMSEs for '{self.target}': {rmses}")

        return np.mean(rmses)

    def search(self, n_trials: int) -> None:
        self.study.optimize(self._objective, n_trials=n_trials)

    def train_best_model(self) -> None:
        CONFIG.lgbm.n_estimators = self.study.best_params["n_estimators"]
        CONFIG.lgbm.learning_rate = self.study.best_params["learning_rate"]
        CONFIG.lgbm.max_depth = self.study.best_params["max_depth"]
        CONFIG.lgbm.num_leaves = self.study.best_params["num_leaves"]
        CONFIG.lgbm.reg_alpha = self.study.best_params["reg_alpha"]
        CONFIG.lgbm.reg_lambda = self.study.best_params["reg_lambda"]
        CONFIG.lgbm.min_child_samples = self.study.best_params["min_child_samples"]

        rmses = []
        models = {}
        for fold in range(CONFIG.num_folds):
            X_train, X_eval, y_train, y_eval = self.df_by_folds[fold]

            model = lgb.LGBMRegressor(**CONFIG.lgbm)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_eval)

            rmse = mean_squared_error(y_eval, y_pred, squared=False)
            rmses.append(rmse)

            models[fold] = model

        for ind in range(CONFIG.num_folds):
            pkl.dump(
                models[ind], open(f"{self.path_for_models_storage}/lgbm_model_{self.target}_{ind}.pkl", "wb")
                     )
        for ind, rmse in enumerate(rmses):
            print(f"RMSE for target '{self.target}' of model {ind} is {rmse}")

        pd.DataFrame(data=[rmses],
                     columns=[f"rmse_{i}" for i in range(CONFIG.num_folds)],
                     ).to_csv(f"{self.path_for_models_storage}/LGBM_rmses_{self.target}.csv", index=False)
