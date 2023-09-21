import torch
from torch import nn
import pytorch_lightning as pl
from transformers import AutoTokenizer

import os
import pandas as pd
import numpy as np

from config import CONFIG


class CommonLitDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.full_text_tokens = df['full_text_tokens'].values
        self.labels = df[['content', 'wording']].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tokens = self.full_text_tokens[index]
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        label = torch.tensor(self.labels[index], dtype=torch.float32)
        return input_ids, attention_mask, label

class CommonLitDataModule(pl.LightningDataModule):
    def __init__(self, fold):
        super().__init__()
        prompts_train_df = pd.read_csv(os.path.join(CONFIG.data_dir, 'prompts_train.csv'))
        summaries_train_df = pd.read_csv(os.path.join(CONFIG.data_dir, 'summaries_train.csv'))
        df = prompts_train_df.merge(summaries_train_df, on='prompt_id')
        tokenizer = AutoTokenizer.from_pretrained(CONFIG.backbone_name)
        df['full_text'] = (df['prompt_question'].values + tokenizer.sep_token +
                           #df['prompt_text'].values + tokenizer.sep_token +
                           df['text'].values)
        
        full_text_tokens = []
        for i in range(len(df)):
            tokens = tokenizer.encode_plus(df.loc[i, 'full_text'], **CONFIG.tokenizer)
            full_text_tokens.append({'input_ids': torch.tensor(tokens['input_ids'], dtype=torch.long), 'attention_mask': torch.tensor(tokens['attention_mask'], dtype=torch.long)})
        df['full_text_tokens'] = full_text_tokens

        promt_ids = np.sort(df['prompt_id'].unique())
        self.train_df = df[df['prompt_id'] != promt_ids[fold]].reset_index(drop=True)
        self.val_df = df[df['prompt_id'] == promt_ids[fold]].reset_index(drop=True)

    def train_dataloader(self):
        dataset = CommonLitDataset(self.train_df)
        return torch.utils.data.DataLoader(dataset=dataset, **CONFIG.data_loaders.train_loader)

    def val_dataloader(self):
        dataset = CommonLitDataset(self.val_df)
        return torch.utils.data.DataLoader(dataset=dataset, **CONFIG.data_loaders.val_loader)