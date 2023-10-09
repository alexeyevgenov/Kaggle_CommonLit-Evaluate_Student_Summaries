from box import Box

CONFIG = {"pca_enabled": False,
          "storage": "../feature_generation/results/3",
          "version": "1",
          "n_trials": 100,
          'models_dir': 'models',
          'db_dir': 'DB',
          "num_folds": 4,
          "pca": {"n_components": 22},
          "ridge": {'alpha': 1.0},
          "data": {"targets": ["content", "wording"],
                   },
          }
CONFIG = Box(CONFIG)
