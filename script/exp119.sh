# rye run python run/train.py +experiment=exp119 decoder.hidden_size=128
# rye run python run/train.py +experiment=exp119 decoder.hidden_size=64
# rye run python run/train.py +experiment=exp119 decoder.hidden_size=128 decoder.num_layers=3
rye run python run/train.py +experiment=exp119 decoder.hidden_size=128 decoder.num_layers=3 decoder.dropout=0.5