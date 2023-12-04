# rye run python run/train.py +experiment=exp134 label_type=laplace offset=10 scale=10 epoch=20 
# rye run python run/train.py +experiment=exp134 label_type=laplace offset=10 scale=20 epoch=20 
# rye run python run/train.py +experiment=exp134 label_type=laplace offset=10 scale=30 epoch=20
rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp134 split=fold_0,fold_1,fold_2,fold_3,fold_4
