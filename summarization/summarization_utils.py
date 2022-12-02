import torch
import transformers
import pandas as pd
import datasets

class SummarizationDataModule(torch.utils.data.Dataset):
    """
    
    """
    def __init__(self, dataset_path, tokenizer, max_length=1024, split='train'):
        """
        
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        # load big_patent_data
        self.data = pd.read_csv(dataset_path+split)
        self.text = self.data['description']
        self.labels = self.data['abstract']

    def __getitem__(self, index):
        """
        
        """
        text = self.text[index]
        label = self.labels[index]
        encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                  max_length=self.max_length, return_tensors='pt')
        encoding['labels'] = self.tokenizer(label, truncation=True, padding='max_length',
                                            max_length=self.max_length, return_tensors='pt')['input_ids']
        return encoding
    def __len__(self):
        """
        
        """
        return len(self.text)