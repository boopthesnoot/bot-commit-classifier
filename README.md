# Bot Commit Classifier
A model that determines whether commit was made by a bot based only on a commit message.

The model that was used in this work was a pre-trained DistilBERT. As model weights already encode a lot of information about natural languages, it takes much less time to fine-tune the model for our specific use case. It has 40% less parameters and is 60% faster than BERT while retaining 97% of its accuracy. 

Data was prepocessed using the following steps: linebreaks and emojis were removed, GitHub hashes were substituted with a `<HASH>` string, commits that have any Chinese/Japanese characters were removed completely.

The DistilBERT model has its own tokenizer that was used with it. The architecture of the model is basically a DistilBERT encoder and a linear layer over it for classification.

PyTorch Lightning and Transformers libraries were used for model and dataloader architectures, wandb.ai was used for logging and tracking the training progress.

Since classes were balanced (it was checked in the `notebooks/preprocessing_test.ipynb`), **accuracy** was chosen as a relevant metric for this task.

Here we can look at the training process: https://wandb.ai/hychin/bot-commit-classifier/runs/1ewsw0yy

## Installation

Clone this repo, change directory to the `bot-commit-classifier` directory:

```bash
git clone https://github.com/boopthesnoot/bot-commit-classifier.git
cd bot-commit-classifier
```

To install the necessary dependencies, run

```bash
conda env create -f environment.yml
conda activate bot-commit-classifier
```

To obtain the full training dataset, download the following file and unzip it into `data/raw`: https://zenodo.org/record/4042126#.Y17zRezMK3J

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

For example, train on a toy dataset using CPU:

```
python src/main.py --mode train --accelerator cpu
```

Train on a full dataset using GPU:
```
python src/main.py --mode train --dataset data/raw/bot-nonbot-msgs.csv --accelerator cuda
```

### Inference

Run inference on a toy dataset (first you need to train the model and specify path to checkpoint in the flag `--model`): the input file should consist of commit messages, one per line.

```
python src/train.py --mode inference --dataset data/processed/predict_sample_small.txt  --model checkpoints/best.ckpt --accelerator cuda
```

### Contacts

Mikhail Lebedev: lebedev_mikhail@outlook.com
