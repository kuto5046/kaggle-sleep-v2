rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp104 scheduler.num_warmup_steps=100 split=fold_0,fold_1,fold_2,fold_3,fold_4
rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp105 scheduler.num_warmup_steps=100 split=fold_0,fold_1,fold_2,fold_3,fold_4

