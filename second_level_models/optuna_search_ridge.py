import os
import optuna
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from config import CONFIG
from utils.data_processing import split_data_on_train_test
import pickle as pkl
import pandas as pd
import wandb
wandb.init(project="visualize-sklearn")

# wandb.login(key="76ca0651b6bb46110bedfc5f63923880b9ee2507")


class Study:
    def __init__(self, data, target):
        self.target = target
        self.data = data
        self.models = {}
        self.rmses = []
        self.path_for_models_storage = f"{CONFIG.models_dir}/{CONFIG.version}"
        os.makedirs(self.path_for_models_storage, exist_ok=True)

        self.df_by_folds = self.prepare_data_for_study()

        storage_for_db = os.path.join(CONFIG.db_dir, CONFIG.version)
        os.makedirs(storage_for_db, exist_ok=True)
        storage = optuna.storages.RDBStorage(url=f"sqlite:///{os.path.join(storage_for_db, f'ridge_{self.target}.db')}")

        self.study = optuna.create_study(direction='minimize',
                                         study_name=CONFIG.version,
                                         storage=storage,
                                         load_if_exists=True)

    def prepare_data_for_study(self) -> dict:
        df_by_folds = {}
        for fold in range(CONFIG.num_folds):
            X_train, X_eval, y_train, y_eval = split_data_on_train_test(self.data, fold, self.target)
            if CONFIG.pca_enabled:
                reducer = PCA(n_components=CONFIG.pca.n_components)
                X_train = reducer.fit_transform(X_train)
                X_eval = reducer.transform(X_eval)
            df_by_folds[fold] = X_train, X_eval, y_train, y_eval
        return df_by_folds

    def _objective(self, trial):
        CONFIG.ridge.alpha = trial.suggest_float('alpha', 0.001, 100)

        rmses = []
        for fold in range(CONFIG.num_folds):
            X_train, X_eval, y_train, y_eval = self.df_by_folds[fold]
            model = Ridge(**CONFIG.ridge)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_eval)

            rmse = mean_squared_error(y_eval, y_pred, squared=False)
            rmses.append(rmse)

        print(f"RMSEs for '{self.target}': {rmses}")

        return np.mean(rmses)

    def search(self, n_trials: int) -> None:
        self.study.optimize(self._objective, n_trials=n_trials)

    def train_best_model(self) -> None:
        CONFIG.ridge.alpha = self.study.best_params["alpha"]

        rmses = []
        models = {}
        for fold in range(CONFIG.num_folds):
            X_train, X_eval, y_train, y_eval = self.df_by_folds[fold]

            model = Ridge(**CONFIG.ridge)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_eval)

            rmse = mean_squared_error(y_eval, y_pred, squared=False)
            rmses.append(rmse)

            models[fold] = model

            wandb.sklearn.plot_regressor(model, X_train, X_eval, y_train, y_eval, model_name='Ridge')

            wandb.finish()

        for ind in range(CONFIG.num_folds):
            pkl.dump(
                models[ind], open(f"{self.path_for_models_storage}/ridge_model_{self.target}_{ind}.pkl", "wb")
                     )
        for ind, rmse in enumerate(rmses):
            print(f"RMSE for target '{self.target}' of model {ind} is {rmse}")

        pd.DataFrame(data=[rmses],
                     columns=[f"rmse_{i}" for i in range(CONFIG.num_folds)],
                     ).to_csv(f"{self.path_for_models_storage}/RIDGE_rmses_{self.target}.csv", index=False)
