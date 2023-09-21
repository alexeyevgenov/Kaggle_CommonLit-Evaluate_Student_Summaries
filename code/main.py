import warnings
import optuna_search


def main():
    warnings.filterwarnings("ignore")
    study = optuna_search.Study()
    study.search(n_trials=20)
    study.train_best_model()


if __name__ == '__main__':
    main()
    # import torch
    #
    # print(torch.backends.cudnn.enabled)
