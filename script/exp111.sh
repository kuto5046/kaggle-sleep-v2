rye run python run/train.py +experiment=exp111 decoder=TransformerDecoder decoder.num_layers=2 decoder.dropout=0.5
rye run python run/train.py +experiment=exp111 decoder=MLPDecoder
rye run python run/train.py +experiment=exp111 decoder=LSTMDecoder
rye run python run/train.py +experiment=exp111 decoder=CNN1DLSTMDecoder