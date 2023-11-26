# rye run python run/train.py +experiment=exp111 decoder=TransformerDecoder decoder.num_layers=2 decoder.dropout=0.5
# rye run python run/train.py +experiment=exp111 decoder=MLPDecoder
# rye run python run/train.py +experiment=exp111 decoder=LSTMDecoder
# rye run python run/train.py +experiment=exp111 decoder=CNN1DLSTMDecoder
# rye run python run/train.py +experiment=exp111 decoder=LSTMDecoder decoder.dropout=0.7
# rye run python run/train.py +experiment=exp111 decoder=LSTMDecoder decoder.num_layers=1

rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp111 decoder=LSTMDecoder split=fold_0,fold_1,fold_2,fold_3,fold_4 model.encoder_name=resnet18
rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp112 decoder=LSTMDecoder split=fold_0,fold_1,fold_2,fold_3,fold_4 model.encoder_name=timm-efficientnet-b2