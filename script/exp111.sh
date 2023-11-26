rye run python run/train.py +experiment=exp111 decoder.name=TransformerDecoder decoder.num_layers=2 decoder.dropout=0.5
rye run python run/train.py +experiment=exp111 decoder.name=MLPDecoder
rye run python run/train.py +experiment=exp111 decoder.name=LSTMDecoder
rye run python run/train.py +experiment=exp111 decoder.name=CNN1DLSTMDecoder