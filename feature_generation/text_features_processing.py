import argparse
from config import CONFIG
from utils.data_processing import split_data_on_train_test, Preprocessor


TEST = True


def prepare_data_for_study(data, target) -> dict:
    df_by_folds = {}
    for fold in range(CONFIG.num_folds):
        X_train, X_eval, y_train, y_eval = split_data_on_train_test(data, fold, target)
        df_by_folds[fold] = X_train, X_eval, y_train, y_eval
    return df_by_folds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("init_folder", type=str, help="path to initial data")
    parser.add_argument("result_folder", type=str, help="path to save processed data")
    arguments = parser.parse_args()

    preprocessor = Preprocessor(TEST, arguments.init_folder, arguments.result_folder)
    preprocessor.run_modin()
