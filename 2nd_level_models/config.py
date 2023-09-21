from box import Box

CONFIG = {"storage": "../feature_generation/results/1",
          "version": "1",
          "n_trials": 100,
          'models_dir': 'models',
          'db_dir': 'DB',
          "num_folds": 4,
          "lgbm": {'objective': 'regression',
                   'metric': 'mse',
                   'boosting_type': 'gbdt',
                   'n_estimators': 100,
                   'learning_rate': 0.01,
                   'max_depth': 3,
                   'num_leaves': 3,
                   'reg_alpha': 0.0,
                   'reg_lambda': 0.0,
                   'min_child_samples': 20,
                   'random_state': 42,
                   },
          "data": {"targets": ["content", "wording"],
                   },
          }
CONFIG = Box(CONFIG)
