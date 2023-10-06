import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List


def feat_to_feat(df: pd.DataFrame, target: List):
    corr_matrix = df.drop(columns=target).corr()
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Feature on feature correlation matrix')
    plt.show()


def feat_to_target(df: pd.DataFrame, target: List):
    corr_matrix = df.corr()
    corr_matrix = corr_matrix.loc[target]
    fig, ax = plt.subplots(figsize=(20, 4))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Feature on target correlation matrix')
    plt.show()
