rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp104 epoch=20 scheduler.num_warmup_steps=100
rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp104 epoch=20 scheduler.num_warmup_steps=300
rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp104 epoch=20 scheduler.num_warmup_steps=600
rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp104 epoch=20 scheduler.num_warmup_steps=1200

rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp105 epoch=20 scheduler.num_warmup_steps=100
rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp105 epoch=20 scheduler.num_warmup_steps=300
rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp105 epoch=20 scheduler.num_warmup_steps=600
rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp105 epoch=20 scheduler.num_warmup_steps=1200

