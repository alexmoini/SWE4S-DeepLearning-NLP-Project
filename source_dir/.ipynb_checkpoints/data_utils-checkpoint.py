from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import AutoTokenizer
import numpy as np
import datasets
import time

class TextSimilarityPairs(Dataset):
    """
    Dataset to load and prepare the phrase dataset data for the model
    """
    def __init__(self, path, tokenizer, sequence_length=10):
        """
        Path is the path to the dataset
        Tokenizer is the tokenizer to use for the dataset
        """
        self.data = pd.read_csv(path, header=0)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length

    def __getitem__(self, index):
        """
        Returns a single item from the dataset, the item is the anchor, target and label
        """
        anchor = self.data.iloc[index]['anchor']
        target = self.data.iloc[index]['target']
        label = self.data.iloc[index]['score']
        label = np.float32(label)
        target_tokens = self.tokenizer(target, padding='max_length', max_length=self.sequence_length,
                                       truncation=True, return_tensors="pt")
        anchor_tokens = self.tokenizer(anchor, padding='max_length', max_length=self.sequence_length,
                                       truncation=True, return_tensors="pt")
        for key in target_tokens.keys():
            target_tokens[key] = target_tokens[key].squeeze(0)
        for key in anchor_tokens.keys():
            anchor_tokens[key] = anchor_tokens[key].squeeze(0)
        
        return anchor_tokens, target_tokens, label

    def __len__(self):
        """
        Returns length of dataset
        """
        return len(self.data)

class CorpusMaskingDataset(Dataset):
    """
    Dataset to prepare and load a corpus of patents
    """
    def __init__(self, path, tokenizer, pre_masked=False, mask_prob=0.15, sequence_length=512, seed=42):
        """
        Path is the path to the csv file containing the corpus
        Tokenizer is the tokenizer to use to tokenize the corpus
        """
        self.data = pd.read_csv(path, header=0)
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.pre_masked = pre_masked
        self.sequence_length = sequence_length
        self.seed = seed

    def __random_masking__(self, tokens, mask_prob):
        """
        Randomly masks the tokens with the mask probability
        """
        np.random.seed(self.seed)
        masked_tokens = tokens.detach().clone()
        for i in range(len(masked_tokens)):
            if np.random.rand() < mask_prob:
                if masked_tokens[i] == 102 or 101:
                    # WE DO NOT WANT TO MASK THE CLS OR SEP TOKENS
                    continue
                else:
                    masked_tokens[i] = 103
        return masked_tokens

    def __getitem__(self, index):
        """
        Returns a dictionary containing the input ids, attention mask and token type ids
        """
        if self.pre_masked: # If the data is already masked
            groundtruth_sequence = self.data.iloc[index]['groundtruth_sequence']
            masked_sequence = self.data.iloc[index]['masked_sequence']

            masked_sequence_tokens = self.tokenizer(masked_sequence, padding='max_length', max_length=self.sequence_length,
                                                    truncation=True, return_tensors="pt")
            groundtruth_sequence_tokens = self.tokenizer(groundtruth_sequence, padding='max_length', max_length=self.sequence_length,
                                                            truncation=True, return_tensors="pt")
            for key in masked_sequence_tokens.keys():
                masked_sequence_tokens[key] = masked_sequence_tokens[key].squeeze(0)
            for key in groundtruth_sequence_tokens.keys():
                groundtruth_sequence_tokens[key] = groundtruth_sequence_tokens[key].squeeze(0)
            return masked_sequence_tokens, groundtruth_sequence_tokens
        else: # We need to mask the sentence
            sequence = self.data.iloc[index]['input_sequence']
            groundtruth_sequence_tokens = self.tokenizer(sequence, padding='max_length', max_length=self.sequence_length,
                                                    truncation=True, return_tensors="pt")
            for key in groundtruth_sequence.keys():
                groundtruth_sequence[key] = groundtruth_sequence[key].squeeze(0)
            masked_sequence_tokens = self.__random_masking__(groundtruth_sequence['input_ids'], self.mask_prob)

            return masked_sequence_tokens, groundtruth_sequence_tokens
    def __len__(self):
        """
        Returns length of dataset
        """
        return len(self.data)