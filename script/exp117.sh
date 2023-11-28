# rye run python run/train.py +experiment=exp117 model.encoder_name=resnet18
# rye run python run/train.py +experiment=exp117 model.encoder_name=resnet34
# rye run python run/train.py +experiment=exp117 model.encoder_name=resnet50
# rye run python run/train.py +experiment=exp117 model.encoder_name=timm-efficientnet-b2
# rye run python run/train.py +experiment=exp117 model.encoder_name=timm-efficientnet-b2 model.encoder_weights=advprop
# rye run python run/train.py +experiment=exp117 model.encoder_name=timm-efficientnet-b2 model.encoder_weights=noisy-student
rye run python run/train.py +experiment=exp117 scheduler.warmup_step_rate=0.05