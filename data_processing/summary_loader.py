import datasets
from nltk import sent_tokenize
import boto3
import time
import argparse
import nltk
nltk.download('punkt')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--output_dir', type=str, default='s3://deeplearning-nlp-bucket/summary_patent_data/')
    parser.add_argument('--dataset_name', type=str, default='big_patent')
    parser.add_argument('--dataset_config_name', type=str, default='d')
    args = parser.parse_args()
    return args

def load_and_split(args):
    print("Loading dataset")
    start = time.time()
    dataset = datasets.load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)
    end = time.time()
    print("Loaded dataset in {} seconds".format(end-start))
    # upload dataset as csv
    s3 = datasets.filesystems.S3FileSystem(anon=True)
    dataset.save_to_disk(args.output_dir+args.split, fs=s3)
if __name__ == '__main__':
    args = get_args()
    load_and_split(args)