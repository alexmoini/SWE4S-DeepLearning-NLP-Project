import datasets
from nltk import sent_tokenize
import boto3
import time
import argparse
import nltk
nltk.download('punkt')
# get S3 role

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    return args

def load_and_split(split):
    print("Loading dataset")
    start = time.time()
    dataset = datasets.load_dataset('big_patent', 'd', split=split)
    end = time.time()
    print("Loaded dataset in {} seconds".format(end-start))
    mlm_set = []
    for sample in dataset['description']:
        sentences = sent_tokenize(sample)
        for sentence in sentences:
            if len(sentence) > 50:
                mlm_set.append(sentence)
    # convert mlm_set to csv with \n delim
    mlm_set = '\n'.join(mlm_set)
    

    # write to S3
    s3 = boto3.resource('s3')
    s3.Bucket('mlm-demo-bucket').put_object(Key=f'big_patent_textiles_{split}.csv', Body=mlm_set)

if __name__ == '__main__':
    args = get_args()
    load_and_split(args['split'])