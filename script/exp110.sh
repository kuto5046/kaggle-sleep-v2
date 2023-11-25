# rye run python run/train.py hydra.mode=MULTIRUN +experiment=exp110 split=fold_0,fold_1,fold_2,fold_3,fold_4

# rye run python run/train.py +experiment=exp110 epoch=20 model.encoder_name=timm-efficientnet-b1
# rye run python run/train.py +experiment=exp110 epoch=20 model.encoder_name=timm-efficientnet-b2
# rye run python run/train.py +experiment=exp110 epoch=20 model.encoder_name=timm-efficientnet-b3
# rye run python run/train.py +experiment=exp110 epoch=20 model.encoder_name=timm-efficientnet-b2 feature_extractor.fmin=20 feature_extractor.fmax=500 feature_extractor.dj=0.072
# rye run python run/train.py +experiment=exp110 epoch=20 model.encoder_name=timm-efficientnet-b2 feature_extractor.fmin=10 feature_extractor.fmax=500 feature_extractor.dj=0.089
# rye run python run/train.py +experiment=exp110 epoch=20 model.encoder_name=timm-efficientnet-b2 feature_extractor.fmin=30 feature_extractor.fmax=500 feature_extractor.dj=0.063


