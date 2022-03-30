"""
python3 -m scripts.make_datamodule \
        +dataset=temporal_analysis \
"""

import hydra

from src.utils import deepconvert
from src.data.datamodule import ParsedDataModule


@hydra.main(config_path='../config', config_name='config.yaml')
def main(cfg):
    cfg = deepconvert(cfg)

    data_module = ParsedDataModule.load_or_create(cfg['dataset'], cfg['cache_dir'])

    print(f"Train dl {data_module.len_train_ds} long")

    train_dl = data_module.train_dataloader()
    if isinstance(train_dl, list):
        train_dl = train_dl[0]
    for data in train_dl:
        break
    import pdb; pdb.set_trace()

    print(f'Valid dl {data_module.len_val_ds} long')

    val_dl = data_module.val_dataloader()
    if isinstance(val_dl, list):
        val_dl = val_dl[0]
    for data in val_dl:
        break
    import pdb; pdb.set_trace()

    print(f'Test dl {data_module.len_test_ds} long')

    test_dl = data_module.test_dataloader()
    if isinstance(test_dl, list):
        test_dl = test_dl[0]
    for data in test_dl:
        break
    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
