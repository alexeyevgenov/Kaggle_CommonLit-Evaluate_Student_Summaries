from box import Box

CONFIG = {"storage": "results",
          "feat_coll_thresh": 0.8,
          "init_data_storage": "folds with embeddings",
          "version": "3",
          "num_folds": 4,
          "data": {"targets": ["content", "wording"],
                   },
          }
CONFIG = Box(CONFIG)
