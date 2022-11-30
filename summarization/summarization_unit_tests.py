from summarization_utils import SummarizationDataModule
from summarization_tuner import SummarizationModel
import unittest
import transformers
import torch

class TestSummarizationDataModule(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.max_length = 1024
        self.dataset_path = 'summarization_test_data/5_rows'
        self.split = ''
        self.data = SummarizationDataModule(self.dataset_path, self.tokenizer, self.max_length, self.split)
    def test_getitem(self):
        """
        Test that the getitem function returns a dictionary with the correct keys
        """
        item = self.data[0]
        self.assertEqual(item.keys(), {'input_ids', 'attention_mask', 'labels'})
        self.assertIsInstance(item['input_ids'], torch.Tensor)
        self.assertIsInstance(item['attention_mask'], torch.Tensor)
        self.assertIsInstance(item['labels'], torch.Tensor)
    def test_len(self):
        """
        Test that the length of the dataset is correct
        """
        self.assertEqual(len(self.data), 5)

class TestSummarizationModel(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case
        """
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
        self.max_length = 1024
        self.dataset_path = 'summarization_test_data/5_rows'
        self.data = SummarizationDataModule(self.dataset_path, self.tokenizer, self.max_length, '')
        self.model = SummarizationModel('facebook/bart-large-cnn')
    def test_forward(self):
        """
        Test that the forward function returns a dictionary with the correct keys
        """
        item = self.data[0]
        output = self.model(item['input_ids'], item['attention_mask'], item['labels'])
        self.assertIsInstance(output['loss'], torch.Tensor)
        self.assertIsInstance(output['logits'], torch.Tensor)
    def test_training_step(self):
        """
        Test that the training step function returns a dictionary with the correct keys
        """
        item = self.data[0]
        output = self.model.training_step(item, 0)
        # output should be torch tensor with gradient
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(output.requires_grad)
    def test_validation_step(self):
        """
        Test that the validation step function returns a dictionary with the correct keys
        """
        item = self.data[0]
        output = self.model.validation_step(item, 0)
        # output should be torch tensor without gradient
        self.assertIsInstance(output, torch.Tensor)
        self.assertFalse(output.requires_grad)
    def test_configure_optimizers(self):
        """
        Test that the configure optimizers function returns a dictionary with the correct keys
        """
        output = self.model.configure_optimizers()
        self.assertEqual(output.keys(), {'optimizer', 'lr_scheduler'})
        self.assertIsInstance(output['optimizer'], torch.optim.Optimizer)
        self.assertIsInstance(output['lr_scheduler']['monitor'], str)
    def test_predict(self):
        """
        Test that the predict function returns a dictionary with the correct keys
        """
        tokens = self.data[0]['input_ids'].tolist()
        text = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        output = self.model.predict(text[0])
        self.assertIsInstance(output, list)
    def test_save_load_model(self):
        """
        Test that the save and load model functions work
        """
        self.model.save_model('summarization_test_data/test_model')
        self.bart = SummarizationModel('facebook/bart-large-cnn', 
                                       checkpoint_dir='summarization_test_data/test_model')

if __name__ == '__main__':
    unittest.main()
