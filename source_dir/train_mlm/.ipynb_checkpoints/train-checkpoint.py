import sys

from backbone_tuners import MaskedLanguageModelingModel
from embedding_backbone import TextEmbeddingBackbone
from data_utils import CorpusMaskingDataset
import argparse
import os
from torch.utils.data import DataLoader, Dataset
import logging
import os
import pytorch_lightning as pl
import transformers

LOGGER = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-architecture", type=str, default="bert-base-uncased")
    parser.add_argument("--model-dir", type=str, default=None) #os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train-dataset", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--val-dataset", type=str, default=os.environ.get("SM_CHANNEL_VALID"))
    parser.add_argument("--output-dir", type=str, default='opt/ml/output/model/trained_model')
    parser.add_argument("--logs-dir", type=str, default='opt/ml/output/logs')
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--masked-lm-prob", type=float, default=0.15)
    args, _ = parser.parse_known_args()
    return args
    
def load_dataset(dataset_dir, tokenizer, max_seq_length):
    dataset = CorpusMaskingDataset(dataset_dir, tokenizer, max_seq_length)
    return dataset
def load_embedding_backbone(model_architecture, model_dir):
    embedding_backbone = TextEmbeddingBackbone(model_architecture, 
                                               checkpoint_dir=model_dir,
                                               pool_outputs=True)
    return embedding_backbone

def load_mlm_model(embedding_backbone):
    mlm_model = MaskedLanguageModelingModel(embedding_backbone)
    return mlm_model

def main():
    args = get_args()
    pl.seed_everything(args.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_architecture)
    embedding_backbone = load_embedding_backbone(args.model_architecture, args.model_dir)
    mlm_model = load_mlm_model(embedding_backbone)
    train_dataset = load_dataset(args.train_dataset, tokenizer, args.max_seq_length)
    val_dataset = load_dataset(args.val_dataset, tokenizer, args.max_seq_length)
    args.batch_size = args.batch_size * max(1, args.gpus)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    if args.gpus > 1:
        trainer = pl.Trainer(devices=args.gpus, 
                             accelerator='gpu',
                             strategy='ddp',
                             max_epochs=args.num_epochs,
                             logger=pl.loggers.TensorBoardLogger(args.logs_dir))
    else:
        if args.gpus == 0:
            raise ValueError("No GPUs found. Please set gpus > 0")
        trainer = pl.Trainer(devices=args.gpus,
                             accelerator='gpu',
                             max_epochs=args.num_epochs,
                             logger=pl.loggers.TensorBoardLogger(args.logs_dir))

    trainer.fit(mlm_model, train_dataloader, val_dataloader)
    mlm_model.save_backbone(args.output_dir)

if __name__ == "__main__":
    main()