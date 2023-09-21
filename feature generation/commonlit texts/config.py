from box import Box

CONFIG = {"storage": "results/1",
          "version": "1",
          "num_folds": 4,
          "data": {"targets": ["content", "wording"],
                   },
          }
CONFIG = Box(CONFIG)
