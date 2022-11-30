import torch
import transformers
import pandas as pd
import datasets

class SummarizationDataModule(torch.utils.Dataset):
    """
    
    """
    def __init__(self, tokenizer, max_length=1024, dataset_name='big-patent',dataset_section='f', split='train'):
        """
        
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        # load big_patent_data
        self.data = datasets.load_dataset(dataset_name, dataset_section, split=split, streaming=False)
        self.text = self.data['description']
        self.labels = self.data['abstract']

    def __getitem__(self, index):
        """
        
        """
        text = self.text[index]
        label = self.labels[index]
        encoding = self.tokenizer(text, label, truncation=True, padding='max_length', max_length=self.max_length)
        return encoding
    def __len__(self):
        """
        
        """
        return len(self.text)