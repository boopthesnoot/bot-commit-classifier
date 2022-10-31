import argparse

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import random_split, DataLoader, Dataset
from torch.optim.lr_scheduler import OneCycleLR

from tqdm import tqdm
from transformers import AutoTokenizer, BertModel

import cleaning

transformers.logging.set_verbosity_error()


class CommitDataset(Dataset):
    def __init__(self, df, transform=None, inference=False):
        """
        torch Dataset class for commits
        :param df: dataframe with either 1 or 2 columns: 'commit_message' and 'is_bot' (latter not needed in inference)
        :param transform: data transformation function, not used yet
        :param inference: is dataset used for inference?
        """
        self.df = df
        self.transform = transform
        self.inference = inference

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        commit_message = self.df.iloc[idx, 0]
        if self.inference:
            return {"commit_message": commit_message}
        is_bot = self.df.iloc[idx, 1]
        sample = {"commit_message": commit_message, "is_bot": is_bot}

        if self.transform:
            sample = self.transform(sample)

        return sample


class CommitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path_to_dataset,
        batch_size=16,
        val_split=0.15,
    ):
        super().__init__()
        self.train_size = 0.7
        self.batch_size = batch_size
        self.path_to_dataset = path_to_dataset
        self.val_split = val_split
        self.max_seq_length = 512
        self.df = None
        self.dataset = None
        self.train_dataset = None
        self.train_len = None
        self.val_dataset = None
        self.test_dataset = None

    def __len__(self):
        """Need that for OneCycleLR learning rate scheduler"""
        return self.train_len

    def prepare_data(self, raw=True):
        """
        Function for cleaning and loading dataframes
        :param raw: True if we use raw data, False if the input data is clean
        :return: None
        """
        if raw:
            self.df = pd.read_csv(
                self.path_to_dataset,
                sep=";",
                usecols=[2, 3],
                names=["commit_message", "is_bot"],
                encoding_errors="ignore",
                dtype=str,
                quoting=3,
            )
            # Cleaning data:
            self.df["commit_message"] = [
                cleaning.clean_text(s) for s in tqdm(self.df["commit_message"])
            ]
            self.df.drop_duplicates(subset="commit_message", inplace=True)
        else:
            self.df = pd.read_csv(
                self.path_to_dataset,
                names=["commit_message", "is_bot"],
            )
        self.df = self.df.dropna()
        self.df["is_bot"] = [0 if i == "BOT" else 1 for i in self.df["is_bot"]]

    def setup(self, stage=None):
        """
        Creating torch Dataset objects, splitting into train/test/val
        :param stage: Not used
        :return: None
        """
        dataset_size = self.df.shape[0]
        self.dataset = CommitDataset(self.df)
        self.train_len = int(self.train_size * dataset_size)
        test_size = (dataset_size - self.train_len) // 2
        val_size = dataset_size - self.train_len - test_size
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset,
        ) = torch.utils.data.random_split(
            self.dataset,
            [self.train_len, val_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


class BertMNLIFinetuner(pl.LightningModule):
    def __init__(self, total_steps=0):
        super().__init__()

        self.bert = BertModel.from_pretrained(
            "distilbert-base-uncased", output_attentions=True
        )
        self.num_classes = 2
        self.W = nn.Linear(self.bert.config.hidden_size, self.num_classes)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.total_steps = total_steps
        self.lr = 4e-5

    def get_device(self):
        return self.bert.state_dict()[
            "encoder.layer.1.attention.self.query.weight"
        ].device

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.998),
            eps=1e-08,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=self.total_steps
        )
        lr_dict = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [lr_dict]

    def forward(self, batch):
        encoded_batch = self.tokenizer.batch_encode_plus(
            batch["commit_message"], max_length=512, padding="max_length"
        )
        input_ids = torch.tensor(encoded_batch["input_ids"]).to(self.get_device())
        attention_mask = torch.tensor(encoded_batch["attention_mask"]).to(
            self.get_device()
        )

        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        h, _, attn = self.bert(**model_inputs)
        out = self.bert(**model_inputs)
        h_cls = out.last_hidden_state[:, 0]
        logits = self.W(h_cls)
        return logits

    def _calculate_loss(self, batch, mode="train"):
        labels = batch["is_bot"]
        preds = self.forward(batch)
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), labels.view(-1))
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        # Logging
        self.log("%s_loss" % mode, loss)
        self.log("%s_acc" % mode, acc)
        return loss, acc

    def predict_step(self, batch, batch_idx, **kwargs):
        preds = self.forward(batch)
        return preds.argmax(dim=-1)

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="test")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="data/raw/bigger_sample.csv",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        required=True,
        help="Choose the usage mode",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Sets the batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Sets the number of epochs"
    )
    parser.add_argument("--model", help="Path to the trained model")
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Choose the accelerator",
    )
    args = parser.parse_args()

    # Setup model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints",
        save_last=True,
        save_top_k=1,
        mode="min",
    )
    model = BertMNLIFinetuner()
    if args.mode == "inference":
        try:
            print(f"loading checkpoint: {args.model}...")
            model.load_from_checkpoint(checkpoint_path=args.model)
        except AttributeError as e:
            print(f"Failed to load the trained model\n{'*'*30}")
            raise e
        df = pd.read_csv(
            args.dataset,
            names=["commit_message"],
            encoding_errors="ignore",
            dtype=str,
            quoting=3,
        )
        df["original_message"] = df["commit_message"].copy()
        df["commit_message"] = [
            cleaning.clean_text(s) for s in tqdm(df["commit_message"])
        ]
        df.dropna(inplace=True)
        inference_dataloader = DataLoader(
            CommitDataset(df, inference=True), batch_size=args.batch_size
        )
        trainer = pl.Trainer()
        predictions = [
            item
            for sublist in trainer.predict(model, dataloaders=inference_dataloader)
            for item in sublist
        ]
        df["predicted_is_bot"] = ["BOT" if i == 0 else "NON-BOT" for i in predictions]
        df.to_csv("data/results/predictions.csv", index=False)

    if args.mode == "train":

        dm = CommitDataModule(
            path_to_dataset=args.dataset,
            batch_size=args.batch_size,
        )
        dm.prepare_data()
        dm.setup()
        model = BertMNLIFinetuner(total_steps=len(dm.train_dataloader()) * args.epochs)
        wandb_logger = WandbLogger(project="bot-commit-classifier")

        trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback],
            accelerator=args.accelerator,
            devices=1,
        )

        trainer.fit(model, dm)


if __name__ == "__main__":
    main()
