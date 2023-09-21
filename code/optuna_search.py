import optuna

import os
import numpy as np
from multiprocessing import Process, Value

from config import CONFIG
from training import train_model

class Study:
    def __init__(self):
        storage_dir = os.path.join(CONFIG.models_dir, CONFIG.version)
        os.makedirs(storage_dir, exist_ok=True)

        storage = optuna.storages.RDBStorage(url=f"sqlite:///{os.path.join(storage_dir, 'optuna.db')}")
        self.study = optuna.create_study(direction='minimize',
                                         study_name=CONFIG.version,
                                         storage=storage,
                                         load_if_exists=True)

    def _objective(self, trial):
        CONFIG.model.optimizer.params.lr = trial.suggest_categorical('lr', [1.5e-5, 2e-5, 2.5e-5, 3e-5])
        CONFIG.model.scheduler.params.pct_start = trial.suggest_float('pct_start', 0.01, 0.3, step=0.01)
        CONFIG.model.dropout = trial.suggest_categorical('dropout', [0, 0.1, 0.15])
        CONFIG.model.hiddendim_lstm = trial.suggest_categorical('hiddendim_lstm', [128, 256, 384, 512])
        CONFIG.trainer.accumulate_grad_batches = trial.suggest_categorical('accumulate_grad_batches', [16, 24, 32, 48])


        mcrmses = []
        for fold in [0]:
            out_val_mcrmse = Value('d', 0.0)
            # p = Process(target=train_model, kwargs={'fold': fold,
            #                                         'checkpoint_dir': None,
            #                                         'out_val_mcrmse': out_val_mcrmse})
            # p.start()
            # p.join()
            train_model(fold=fold, checkpoint_dir=None, out_val_mcrmse=out_val_mcrmse)
            mcrmses.append(out_val_mcrmse.value)
        return np.mean(mcrmses)

    def search(self, n_trials):
        self.study.optimize(self._objective, n_trials=n_trials)

    def train_best_model(self):
        df = self.study.trials_dataframe(('value', 'params', 'state'))
        df = df[df['state'] == 'COMPLETE']
        df = df.sort_values('value').reset_index(drop=True)
        best_params = df.iloc[0]

        CONFIG.model.optimizer.params.lr = best_params['params_lr']
        CONFIG.model.scheduler.params.pct_start = best_params['params_pct_start']
        CONFIG.model.dropout = best_params['params_dropout']
        CONFIG.trainer.accumulate_grad_batches = int(best_params['params_accumulate_grad_batches'])

        checkpoint_dir = os.path.join(CONFIG.models_dir, CONFIG.version, 'model 0')

        for fold in range(CONFIG.n_splits):
            print(f'{fold=}')
            out_val_mcrmse = Value('d', 0.0)
            p = Process(target=train_model, kwargs={'fold': fold, 
                                                    'checkpoint_dir': checkpoint_dir,
                                                    'out_val_mcrmse': out_val_mcrmse})
            p.start()
            p.join()
