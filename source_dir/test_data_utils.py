import unittest
from data_utils import *
import torch
import transformers

class TestTextSimilarityPairs(unittest.TestCase):
    def setUp(self):
        """
        Setup the test
        1. Loads dataset
        """
        self.path = 'data/1000_pairs.csv'
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        self.sequence_length = 10
        self.dataset = TextSimilarityPairs(self.path, self.tokenizer, self.sequence_length)

    def test_getitem(self):
        """
        Test the getitem method
        1. Check if the anchor, target and label are returned
        2. Check if the anchor and target are tokenized correctly

        """
        anchor_tokens, target_tokens, label = self.dataset[0]
        self.assertEqual(anchor_tokens['input_ids'].shape, torch.Size([self.sequence_length]))
        self.assertEqual(anchor_tokens['token_type_ids'].shape, torch.Size([self.sequence_length]))
        self.assertEqual(anchor_tokens['attention_mask'].shape, torch.Size([self.sequence_length]))
        self.assertEqual(target_tokens['input_ids'].shape, torch.Size([self.sequence_length]))
        self.assertEqual(target_tokens['token_type_ids'].shape, torch.Size([self.sequence_length]))
        self.assertEqual(target_tokens['attention_mask'].shape, torch.Size([self.sequence_length]))
        self.assertEqual(label.shape, torch.Size([1]))

class TestCorpusMaskingDataset(unittest.TestCase):
    """
    Testing object for the CorpusMaskingDataset
    """
    def setUp(self):
        """
        Loads 2 datasets of 1000 sequences each, one with pre-masked sequences and one without
        """
        self.path_not_masked = 'data/1000_sentences_not_masked.csv'
        self.path_masked = 'data/1000_sentences_masked.csv'

        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        self.sequence_length = 128
        self.not_masked_dataset = CorpusMaskingDataset(self.path_not_masked, self.tokenizer,
                                                       self.sequence_length, mask_prob=0.15,
                                                       pre_masked=False)
        self.masked_dataset = CorpusMaskingDataset(self.path_masked, self.tokenizer,
                                                   self.sequence_length, pre_masked=True)

    def test_getitem(self):
        """
        Tests __getitem__ method of the CorpusMaskingDataset
        1. Checks that the output is a dictionary with the correct keys and sizes
        2. Checks that there is always masked tokens in the output
        """
        masked_sentence, groundtruth_sentence = self.dataset[0]
        # Check for correct sequence sizes
        self.assertEqual(masked_sentence['input_ids'].shape, torch.Size([self.sequence_length]))
        self.assertEqual(masked_sentence['token_type_ids'].shape, torch.Size([self.sequence_length]))
        self.assertEqual(masked_sentence['attention_mask'].shape, torch.Size([self.sequence_length]))
        self.assertEqual(groundtruth_sentence['input_ids'].shape, torch.Size([self.sequence_length]))
        self.assertEqual(groundtruth_sentence['token_type_ids'].shape, torch.Size([self.sequence_length]))
        self.assertEqual(groundtruth_sentence['attention_mask'].shape, torch.Size([self.sequence_length]))

        # ensure each masked sentence has a masked token
        self.assertTrue(torch.any(masked_sentence['input_ids'] == self.tokenizer.mask_token_id))


        