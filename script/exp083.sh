rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp083 batch_size=64,128,256
rye run python run/train.py +experiment=exp084
rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp085 split=fold_0,fold_1,fold_2,fold_3,fold_4