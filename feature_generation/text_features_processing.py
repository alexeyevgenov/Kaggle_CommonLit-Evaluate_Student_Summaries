import pandas as pd
import numpy as np
import pickle as pkl
from third_level_models.config import CONFIG
from utils.data_processing import group_folds_in_a_single_df, remove_highly_collinear_variables, \
    split_data_on_train_test, normalize_data, drop_columns

np.random.seed(1)


def prepare_data_for_study(data, target) -> dict:
    df_by_folds = {}
    for fold in range(CONFIG.num_folds):
        X_train, X_eval, y_train, y_eval = split_data_on_train_test(data, fold, target)
#         id_train = X_train["student_id"]
#         id_eval = X_eval["student_id"]
#         X_train = X_train.drop(columns=["student_id"])
#         X_eval = X_eval.drop(columns=["student_id"])
        X_train, X_eval, scaler = normalize_data(X_train, X_eval)
#         X_train = pd.concat([X_train, id_train], axis=1)
#         X_eval = pd.concat([X_eval, id_eval], axis=1)
        df_by_folds[fold] = X_train, X_eval, y_train, y_eval
    return df_by_folds


data = group_folds_in_a_single_df(CONFIG.storage, CONFIG.num_folds)
emb_columns = [el for el in data.columns if "emb_" in el]
fold_nums = data["fold"]
# prompt_id = data["student_id"]
data = data.drop(columns=emb_columns + drop_columns + ["fold"])  # + ["student_id"])
data = remove_highly_collinear_variables(data, CONFIG.feat_coll_thresh)
data = pd.concat([data, fold_nums], axis=1)   # , prompt_id
df_by_folds = prepare_data_for_study(data, CONFIG.data.targets)

pkl.dump(df_by_folds, open(f"ready_text_features/text_features.pkl", "wb"))
