from box import Box

CONFIG = {"storage": "../feature_generation/results/3",
          "version": "3",
          "n_trials": 100,
          'models_dir': 'models',
          'db_dir': 'DB',
          "num_folds": 4,
          "feat_coll_thresh": 0.8,
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
          "svr": {'kernel': 'rbf',
                  'C': 1.0,
                  'epsilon': 0.1,
                  'gamma': 'scale',
                  },
          "ridge": {'alpha': 1.0},
          "data": {"targets": ["content", "wording"],
                   },
          }
CONFIG = Box(CONFIG)
