import torch
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

import os
import gc

from config import CONFIG
from dataset import CommonLitDataModule
from model import CommonLitModel

def train_model(fold, checkpoint_dir, out_val_mcrmse):
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_float32_matmul_precision('medium')
    seed_everything(CONFIG.seed)
    dm = CommonLitDataModule(fold)
    
    CONFIG.model.scheduler.params.steps_per_epoch = len(dm.train_dataloader()) // CONFIG.trainer.accumulate_grad_batches + 1
    model = CommonLitModel()
    # model = torch.compile(model)
    
    lr_monitor_callback = callbacks.LearningRateMonitor()
    trainer_callbacks = [lr_monitor_callback]
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_callback = callbacks.ModelCheckpoint(dirpath=checkpoint_dir, 
                                                        save_weights_only=True,
                                                        filename=f'fold={fold}-' + '{epoch}-{val_mcrmse:.5f}',
                                                        every_n_epochs=1,
                                                        save_top_k=-1)
        trainer_callbacks.append(checkpoint_callback)
    
    trainer = pl.Trainer(
        callbacks=trainer_callbacks,
        **CONFIG.trainer
    )

    trainer.fit(model, dm)
    out_val_mcrmse.value = trainer.callback_metrics['val_mcrmse'].min()

    del model
    del trainer

    gc.collect()
    torch.cuda.empty_cache()
