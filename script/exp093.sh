rye run python run/train.py +experiment=exp093 epoch=20 decoder.attention_window_size=16 decoder.attention_pooling=avg
rye run python run/train.py +experiment=exp093 epoch=20 decoder.attention_window_size=32 decoder.attention_pooling=avg
rye run python run/train.py +experiment=exp093 epoch=20 decoder.attention_window_size=64 decoder.attention_pooling=avg
rye run python run/train.py +experiment=exp093 epoch=20 decoder.attention_window_size=128 decoder.attention_pooling=avg
rye run python run/train.py +experiment=exp093 epoch=20 decoder.attention_window_size=16 decoder.attention_pooling=max
rye run python run/train.py +experiment=exp093 epoch=20 decoder.attention_window_size=32 decoder.attention_pooling=max
rye run python run/train.py +experiment=exp093 epoch=20 decoder.attention_window_size=64 decoder.attention_pooling=max
rye run python run/train.py +experiment=exp093 epoch=20 decoder.attention_window_size=128 decoder.attention_pooling=max
# 32はなぜかスコア低い
# 16 or 128が良さそう
# avg or maxは若干maxだが、あまり差はない