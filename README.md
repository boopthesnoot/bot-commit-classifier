# Bot Commit Classifier
A model that determines whether commit was made by a bot based only on a commit message.

The model that was used in this work was a pre-trained DistilBERT. As model weights already encode a lot of information about natural languages, it takes much less time to fine-tune the model for our specific use case. It has 40% less parameters and is 60% faster than BERT while retaining 97% of its accuracy. 

Data was prepocessed using the following steps: linebreaks and emojis were removed, GitHub hashes were substituted with a `<HASH>` string, commits that have any Chinese/Japanese characters were removed completely.

The DistilBERT model has its own tokenizer that was used with it. The architecture of the model is basically a DistilBERT encoder and a linear layer over it for classification.

PyTorch Lightning and Transformers libraries were used for model and dataloader architectures, wandb.ai was used for logging and tracking the training progress.

Since classes were balanced (it was checked in the `notebooks/preprocessing_test.ipynb`), **accuracy** was chosen as a relevant metric for this task.

Here we can look at the training process: https://wandb.ai/hychin/bot-commit-classifier/runs/1ewsw0yy

## Installation

Instructions below were tested on MacBook Air M1 (8-core CPU, 16GB RAM) and on the cluster (56 CPU cores, 1TB RAM, Nvidia V100)
Clone this repo, change directory to the `bot-commit-classifier` directory:

```bash
git clone https://github.com/boopthesnoot/bot-commit-classifier.git
cd bot-commit-classifier
```

To install the necessary dependencies, run

```bash
conda env create -f environment.yml
conda activate bot-commit-classifier
poetry config experimental.new-installer false && poetry install --no-root
```

To obtain the full training dataset, download the following file and unzip it into `data/raw`: https://zenodo.org/record/4042126

## Usage

### Training

To start training, run `src/main.py` from `bot-commit-classifier` directory with necessary flags 

```
usage: main.py [-h] [--dataset DATASET] --mode {train,inference} [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--model MODEL]
               [--accelerator {cuda,cpu}]

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Path to the dataset
  --mode {train,inference}
                        Choose the usage mode
  --batch_size BATCH_SIZE
                        Sets the batch size
  --epochs EPOCHS       Sets the number of epochs
  --model MODEL         Path to the trained model (for inference)
  --accelerator {cuda,cpu}
                        Choose the accelerator
```

As part of the dependancies were resolved with Poetry, will have to prefix python calls with `poetry run`.

For example, train on a toy dataset using CPU:

```
poetry run python src/main.py --mode train --accelerator cpu
```

Train on a full dataset using GPU:
```
poetry run python src/main.py --mode train --dataset data/raw/bot-nonbot-msgs.csv --accelerator cuda
```

### Inference

Run inference on a toy dataset (first you need to train the model and specify path to checkpoint in the flag `--model`): the input file should consist of commit messages, one per line.

```
poetry run python src/train.py --mode inference --dataset data/processed/predict_sample_small.txt  --model checkpoints/best.ckpt --accelerator cuda
```

## Performance

Since classes were balanced (it was checked in the `notebooks/preprocessing_test.ipynb`), **accuracy** was chosen as a relevant metric for this task. For the loss function, cross enthropy was used. Here we can see the performance of the model during training

<img src="https://github.com/boopthesnoot/bot-commit-classifier/blob/main/data/results/W%26B%20Chart%2031_10_2022%2C%2022_00_07.svg" width="500" height="300">
<img src="https://github.com/boopthesnoot/bot-commit-classifier/blob/main/data/results/W%26B%20Chart%2031_10_2022%2C%2022_15_42.svg" width="500" height="300">


Accuracy on the validation dataset:

<img src="https://github.com/boopthesnoot/bot-commit-classifier/blob/main/data/results/W%26B%20Chart%2031_10_2022%2C%2022_19_20.png" width="500" height="300">

These numbers are a bit concerning, and I will have to double check that there are no data leaks.

### Contacts

Mikhail Lebedev: lebedev_mikhail@outlook.com
