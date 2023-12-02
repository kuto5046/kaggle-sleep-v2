# rye run python run/train.py +experiment=exp133 label_type=laplace offset=10 scale=10
# rye run python run/train.py +experiment=exp133 label_type=laplace offset=10 scale=20
# rye run python run/train.py +experiment=exp133 label_type=laplace offset=10 scale=30
# rye run python run/train.py +experiment=exp133 label_type=laplace offset=10 scale=40
rye run python run/train.py +experiment=exp133 label_type=laplace offset=10 scale=30 notes='laplace label with only event loss'
rye run python run/train.py +experiment=exp133 label_type=gaussian offset=10 sigma=10 notes='gaussian label with only event loss'