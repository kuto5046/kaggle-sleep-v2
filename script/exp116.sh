# rye run python run/train.py +experiment=exp116 downsample_rate=1 epoch=20
# rye run python run/train.py +experiment=exp116 downsample_rate=2 epoch=20
# rye run python run/train.py +experiment=exp116 downsample_rate=4 epoch=20
# rye run python run/train.py +experiment=exp116 downsample_rate=6 epoch=20
# rye run python run/train.py +experiment=exp116 downsample_rate=12 epoch=20
# rye run python run/train.py +experiment=exp116 downsample_rate=4 duration=5760 epoch=20
# rye run python run/train.py +experiment=exp116 downsample_rate=4 duration=17280 epoch=20

rye run python run/train.py +experiment=exp116 downsample_rate=4 duration=17280 epoch=20 augmentation.mixup_prob=0.5
rye run python run/train.py +experiment=exp116 downsample_rate=4 duration=17280 epoch=20 augmentation.cutmix_prob=0.5
rye run python run/train.py +experiment=exp116 downsample_rate=4 duration=17280 epoch=20 augmentation.mixup_prob=0.5 augmentation.cutmix_prob=0.5
rye run python run/train.py +experiment=exp116 downsample_rate=4 duration=17280 epoch=20 augmentation.local_shuffle_prob=0.5 augmentation.local_shuffle_window_size=6