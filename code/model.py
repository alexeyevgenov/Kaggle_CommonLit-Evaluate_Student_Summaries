import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch import optim
# import torch_optimizer
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel
import numpy as np
from sklearn.metrics import mean_squared_error

from config import CONFIG

class LSTMPooling(nn.Module):
    def __init__(self, num_layers, hidden_size, hiddendim_lstm):
        super(LSTMPooling, self).__init__()
        self.num_hidden_layers = num_layers
        self.hidden_size = hidden_size
        self.hiddendim_lstm = hiddendim_lstm
        self.lstm = nn.LSTM(self.hidden_size, self.hiddendim_lstm, batch_first=True)
    
    def forward(self, all_hidden_states):
        ## forward
        hidden_states = torch.stack([all_hidden_states[layer_i][:, 0].squeeze()
                                     for layer_i in range(1, self.num_hidden_layers + 1)], 
                                     dim=-1)
        hidden_states = hidden_states.view(-1, self.num_hidden_layers, self.hidden_size)
        out, _ = self.lstm(hidden_states, None)
        out = out[:, -1, :]
        return out

class CommonLitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self._build_model()
        self._criterion = eval(CONFIG.model.criterion)()

        self.outputs = {
            'train': {
                  'preds': [],
                  'labels': []
             },
             'val': {
                  'preds': [],
                  'labels': []
             }
        }

    def _build_model(self):
        self.backbone = AutoModel.from_pretrained(CONFIG.backbone_name)
        self.backbone_config = AutoConfig.from_pretrained(CONFIG.backbone_name)
        self.pooling = LSTMPooling(self.backbone_config.num_hidden_layers, 
                                   self.backbone_config.hidden_size, 
                                   CONFIG.model.hiddendim_lstm)
        self.fc = nn.Sequential(nn.Dropout(CONFIG.model.dropout),
                                nn.Linear(CONFIG.model.hiddendim_lstm, 2))
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.backbone_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
           
    def forward(self, input_ids, attention_mask):
        x = self.backbone(input_ids=input_ids,
                          attention_mask=attention_mask,
                          output_hidden_states=True)
        x = self.pooling(x.hidden_states)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch, 'train')
        return {'loss': loss, 'preds': preds, 'labels': labels}

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.__share_step(batch, 'val')
        return {'preds': preds, 'labels': labels}

    def __share_step(self, batch, mode):
        input_ids, attention_mask, labels = batch

        preds = self.forward(input_ids, attention_mask).squeeze(1)
        loss = self._criterion(preds, labels)

        preds = preds.detach().cpu()
        labels = labels.detach().cpu()

        self.outputs[mode]['preds'].append(preds)
        self.outputs[mode]['labels'].append(labels)
        return loss, preds, labels

    def MCRMSE(self, y_trues, y_preds):
        scores = []
        idxes = y_trues.shape[1]
        for i in range(idxes):
            y_true = y_trues[:,i]
            y_pred = y_preds[:,i]
            score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
            scores.append(score)
        mcrmse_score = np.mean(scores)
        return mcrmse_score

    def on_training_epoch_end(self):
        self.__share_epoch_end('train')

    def on_validation_epoch_end(self):
        self.__share_epoch_end('val')

    def __share_epoch_end(self, mode):
        preds = [pred for pred in self.outputs[mode]['preds']]
        labels = [label for label in self.outputs[mode]['labels']]
        preds = torch.cat(preds).float().numpy()
        labels = torch.cat(labels).float().numpy()

        mcrmse = self.MCRMSE(labels, preds)
        self.log(f'{mode}_mcrmse', mcrmse)

        self.outputs[mode]['preds'] = []
        self.outputs[mode]['labels'] = []

    def configure_optimizers(self):
        optimizer_params = CONFIG.model.optimizer.params
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        model_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
              'lr': optimizer_params.lr, 'weight_decay': optimizer_params.weight_decay, 'eps': optimizer_params.eps},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
              'lr': optimizer_params.lr, 'weight_decay': 0.0, 'eps': optimizer_params.eps},
        ]
        optimizer = eval(CONFIG.model.optimizer.name)(model_parameters)
        
        scheduler_params = CONFIG.model.scheduler.params
        scheduler_params.max_lr = CONFIG.model.optimizer.params.lr
        scheduler_params.epochs = CONFIG.trainer.max_epochs


        scheduler = eval(CONFIG.model.scheduler.name)(optimizer, **scheduler_params)
        scheduler_config = {
            'scheduler': scheduler,
            'interval': 'step',
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}