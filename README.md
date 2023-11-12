# Child Mind Institute - Detect Sleep States

This repository is for [Child Mind Institute - Detect Sleep States](https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/overview)

## Build Environment
### 1. install [rye](https://github.com/mitsuhiko/rye)

[install documentation](https://rye-up.com/guide/installation/#installing-rye)

MacOS
```zsh
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.zshrc
source ~/.zshrc
```

Linux
```bash
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
source ~/.bashrc
```

Windows  
see [install documentation](https://rye-up.com/guide/installation/)

### 2. Create virtual environment

```bash
rye sync
```

### 3. Activate virtual environment

```bash
. .venv/bin/activate
```

### Set path
Rewrite run/conf/dir/local.yaml to match your environment

```yaml
data_dir: 
processed_dir: 
output_dir: 
model_dir: 
sub_dir: ./
```

## Prepare Data

### 1. Download data

```bash
cd data
kaggle competitions download -c child-mind-institute-detect-sleep-states
unzip child-mind-institute-detect-sleep-states.zip
```

### 2. Preprocess data

```bash
rye run python -m run/prepare_data.py phase=train,test
```

## Train Model
The following commands are for training the model of LB0.714
```bash
rye run python run/train.py downsample_rate=2 duration=5760 exp_name=exp001 batch_size=32
```

You can easily perform experiments by changing the parameters because [hydra](https://hydra.cc/docs/intro/) is used.
The following commands perform experiments with downsample_rate of 2, 4, 6, and 8.

```bash
rye run python run/train.py hydra.mode=MULTIRUN downsample_rate=2,4,6,8
```

複数foldを学習させるとき
```bash
python run/train.py hydra.mode=MULTIRUN +experiment=exp001 split=fold_0,fold_1,fold_2,fold_3,fold_4
```

hydra memo

experimentフォルダにexp001.yamlをtrain.yamlの差分のみ記載しておくと以下のように実行できる
```bash
python run/train.py +experiment=exp001
```

multirunで複数のlist形式のパラメータを指定する場合は以下のように文字列で指定する
```bash
python run/train.py hydra.mode=MULTIRUN +experiment=exp001 'downsample_rate=[2,4,6,8],[2,4]'
```


## Upload 
最初の場合は--newをつける
upload model
```bash
rye run python tools/upload_model.py 
```

upload script
```bash
rye run python tools/upload_code.py
```

## Inference
The following commands are for inference of LB0.714 
```bash
rye run python run/inference.py dir=kaggle exp_name=exp001 weight.run_name=single downsample_rate=2 duration=5760 model.encoder_weights=null post_process.score_th=0.005 post_process.distance=40 phase=test
```

複数foldで推論するとき
multirunを使うとモデルの読み込みがちょっと面倒なので個別で実行する
```bash
python run/inference.py +experiment=exp001 weight.run_name=run0
python run/inference.py +experiment=exp001 weight.run_name=run1
```

アンサンブルやスタッキングをする場合は上記の方法で個別の予測値を作成しそれを利用する