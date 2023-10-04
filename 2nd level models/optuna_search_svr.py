import os
import optuna
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import wandb
from config import CONFIG
from feature_generation.data_processing_unit import normalize_data, split_data_on_train_test
import pickle as pkl
import pandas as pd

# wandb.login(key="76ca0651b6bb46110bedfc5f63923880b9ee2507")


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
        storage = optuna.storages.RDBStorage(url=f"sqlite:///{os.path.join(storage_for_db, f'svr_{self.target}.db')}")

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
                os.makedirs(self.path_for_models_storage, exist_ok=True)
                pkl.dump(
                    scaler, open(f"{self.path_for_models_storage}/scaler_{fold}.pkl", "wb"))

        return df_by_folds

    def _objective(self, trial):
        CONFIG.svr.C = trial.suggest_float('C', 0.01, 1000)
        CONFIG.svr.epsilon = trial.suggest_float('epsilon', 0.01, 0.1)
        # CONFIG.svr.gamma = trial.suggest_float('gamma', 0.001, 10)

        rmses = []
        for fold in range(CONFIG.num_folds):
            X_train, X_eval, y_train, y_eval = self.df_by_folds[fold]
            model = SVR(**CONFIG.svr)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_eval)

            rmse = mean_squared_error(y_eval, y_pred, squared=False)
            rmses.append(rmse)

        print(f"RMSEs for '{self.target}': {rmses}")

        return np.mean(rmses)

    def search(self, n_trials: int) -> None:
        self.study.optimize(self._objective, n_trials=n_trials)

    def train_best_model(self) -> None:
        CONFIG.svr.C = self.study.best_params["C"]
        CONFIG.svr.epsilon = self.study.best_params["epsilon"]
        # CONFIG.svr.gamma = self.study.best_params["gamma"]

        rmses = []
        models = {}
        for fold in range(CONFIG.num_folds):
            X_train, X_eval, y_train, y_eval = self.df_by_folds[fold]

            model = SVR(**CONFIG.svr)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_eval)

            rmse = mean_squared_error(y_eval, y_pred, squared=False)
            rmses.append(rmse)

            models[fold] = model

        for ind in range(CONFIG.num_folds):
            pkl.dump(
                models[ind], open(f"{self.path_for_models_storage}/svr_model_{self.target}_{ind}.pkl", "wb")
                     )
        for ind, rmse in enumerate(rmses):
            print(f"RMSE for target '{self.target}' of model {ind} is {rmse}")

        pd.DataFrame(data=[rmses],
                     columns=[f"rmse_{i}" for i in range(CONFIG.num_folds)],
                     ).to_csv(f"{self.path_for_models_storage}/SVR_rmses_{self.target}.csv", index=False)
