rye run python run/train.py +experiment=exp082 downsample_rate=1 
rye run python run/train.py +experiment=exp082 downsample_rate=2
rye run python run/train.py +experiment=exp082 downsample_rate=3 
rye run python run/train.py +experiment=exp082 downsample_rate=4 epoch=50
rye run python run/train.py +experiment=exp082 downsample_rate=6 epoch=50
rye run python run/train.py +experiment=exp082 downsample_rate=2 epoch=50 decoder=LSTMDecoder