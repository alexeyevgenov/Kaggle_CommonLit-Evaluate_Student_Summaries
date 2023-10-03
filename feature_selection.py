import pandas as pd
from feature_generation.config import CONFIG


drop_columns = ["prompt_length", "prompt_id", "prompt_question", "prompt_title", "prompt_text", "student_id", "text",
                "full_text", "fixed_summary_text", "embeddings", "fold"]

total_df = pd.DataFrame()
for i in range(4):
    fold_df = pd.read_feather("feature_generation/results/2/" + f"/preprocessed fold {i}.ftr")
    fold_df["fold"] = i
    if total_df.empty:
        total_df = fold_df
    else:
        total_df = pd.concat([total_df, fold_df], axis=0, ignore_index=True)

numerical_data = total_df.drop(columns=drop_columns)
# numerical_data.drop(columns=["wording", "content"])
numerical_data

# from sklearn.datasets import load_breast_cancer
# from sklearn.feature_selection import GenericUnivariateSelect, chi2, mutual_info_regression
# X = numerical_data.drop(columns=CONFIG.data.targets)
# y_wording = numerical_data.wording
# y_content = numerical_data.content
# transformer = GenericUnivariateSelect(mutual_info_regression, mode='k_best', param=10)
# X_wording = transformer.fit_transform(X, y_wording)
# X_content = transformer.fit_transform(X, y_content)
# X_content
