# bot-commit-classifier
A model that determines whether commit was made by a bot based only on commit message.

The model that was used in this work was a pre-trained DistilBERT. As model weights already encode a lot of information about natural languages, it takes much less time to fine-tune the model for our specific use case. It has 40% less parameters and is 60% faster than the BERT while retaining 97% of the accuracy. 

Data was prepocessed using following steps: linebreaks and emojis were removed, github hashes were substituted with a string `<HASH>`, entire commits that have any chinese/japanese characters were removed.

The DistilBERT model has its own tokenizer that was used with it. The architecture of the model is basically a DistilBERT encoder and a linear layer over it for classification.

PyTorch Lightning and Transformers libraries were used for model and dataloader architectures, wandb.ai was used for logging and tracking training progress.

As classes were balanced (it was checked in the `notebooks/preprocessing_test.ipynb`, **accuracy** was chosen as a relevant metric for this task.

## Installation:

Clone this repo, change directory to the `bot-commit-classifier` directory:

```bash
git clone https://github.com/boopthesnoot/bot-commit-classifier.git
cd bot-commit-classifier
```

To install the necessary dependancies, run

```bash
conda env create -f environment.yml
conda activate bot-commit-classifier
```

To download full training dataset, download the following file and unzip it into `data/raw`: https://zenodo.org/record/4042126#.Y17zRezMK3J

## Usage:

### Training:

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

### Inference:

Run inference on a toy dataset (first you need to train the model and specify path to checkpoint in the flag `--model`): The input file should consist of commit messages, one per line.

```
python src/train.py --mode inference --dataset data/processed/predict_sample_small.txt  --model checkpoints/best.ckpt --accelerator cuda
```
