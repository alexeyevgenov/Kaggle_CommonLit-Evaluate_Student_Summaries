from box import Box

CONFIG = {'data_dir': r'C:\Users\Acer\Мой диск\ML & DS\DS Competitions\CommonLit - Evaluate Student Summaries\commonlit-evaluate-student-summaries',  #'/home/ilya/Yandex.Disk/Kaggle/CommonLit - Evaluate Student Summaries/Data',
          'models_dir':  'models',  # '/home/ilya/Yandex.Disk/Kaggle/CommonLit - Evaluate Student Summaries/Models',
          'logs_dir':  'logs',  #'/home/ilya/Yandex.Disk/Kaggle/CommonLit - Evaluate Student Summaries/Logs',
          'version': '1.0.3.3.2',
          'seed': 1999 + 1996,
          'backbone_name': 'microsoft/deberta-v3-base',
          'n_splits': 4,
          'tokenizer': {
              'truncation': True,
              'add_special_tokens': True,
              'max_length': 1024,
              'padding': 'max_length'
          },
          'data_loaders': {
              'train_loader':{
                  'batch_size': 2,
                  'shuffle': True,
                  'num_workers': 2,
                  'pin_memory': False,
                  'drop_last': True,
              },
              'val_loader': {
                  'batch_size': 2,
                  'shuffle': False,
                  'num_workers': 2,
                  'pin_memory': False,
                  'drop_last': False
               }
          },
          'model': {
              'dropout': 0,
              'criterion': 'nn.SmoothL1Loss',
              'hiddendim_lstm': 256,
              'optimizer':{
                  'name': 'optim.AdamW',
                  'params':{
                      'lr': None,  #1.5e-5,
                      'eps': 1e-6,
                      'weight_decay': 1e-2,
                  }
              },
              'scheduler':{
                  'name': 'optim.lr_scheduler.OneCycleLR',
                  'params':{
                      'max_lr': None,
                      'pct_start': None,
                      'steps_per_epoch': None,
                      'epochs': None,
                      'cycle_momentum': False,
                      'div_factor': 1e4,
                      'anneal_strategy': 'linear',
                      #'verbose': True,
                  }
              }
          },
          'trainer': {
              'max_epochs': 5,
              'gradient_clip_val': 1.0,
              'accumulate_grad_batches': 16,
              'num_sanity_val_steps': 0,
              'accelerator': 'cuda',
              'precision': '16-mixed',
              'enable_progress_bar': True,
              'enable_model_summary': True,
              # 'deterministic': True,
              # 'benchmark': False,
              # 'limit_train_batches': 1
          }
}

CONFIG = Box(CONFIG)
