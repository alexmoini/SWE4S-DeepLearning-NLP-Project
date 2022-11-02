import pytorch_lightning as pl
import torch
import transformers
from custom_losses import CosineEmbeddingLoss
from custom_losses import cosine_similarity
import pandas as pd

class TextSimilarityModel(pl.LightningModule):
    """
    Text similarity model
    """
    def __init__(self, embedding_backbone, learning_rate=1e-3, loss=CosineEmbeddingLoss(), prediction_example=None, tokenizer=None):
        """
        Model name can be any of the models from the transformers library
        Freeze is a boolean to freeze the weights of the model
        Pool outputs is a boolean to pool the outputs of the model
        Learning rate is the learning rate for the optimizer
        """
        super().__init__()
        self.model = embedding_backbone
        self.loss = loss
        self.learning_rate = learning_rate
        self.prediction_example = prediction_example # should be DF with columns: anchor, target, label
        self.tokenizer = tokenizer # should be transformers tokenizer
        self.epoch_num = 0

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model
        """
        return self.model(input_ids, attention_mask)

    def training_step(self, batch, batch_idx):
        """
        Training step of the model
        """
        anchor_tokens, target_tokens, labels = batch
        anchor_input_ids, anchor_attention_mask = anchor_tokens['input_ids'], anchor_tokens['attention_mask']
        target_input_ids, target_attention_mask = target_tokens['input_ids'], target_tokens['attention_mask']
        
        anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)
        target_embeddings = self.model(target_input_ids, target_attention_mask)
        loss = self.loss(anchor_embeddings, target_embeddings, labels)
        self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model
        """
        anchor_tokens, target_tokens, labels = batch
        anchor_input_ids, anchor_attention_mask = anchor_tokens['input_ids'], anchor_tokens['attention_mask']
        target_input_ids, target_attention_mask = target_tokens['input_ids'], target_tokens['attention_mask']
        
        anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)
        target_embeddings = self.model(target_input_ids, target_attention_mask)
        loss = self.loss(anchor_embeddings, target_embeddings, labels)
        self.log('valid_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        torch.cuda.empty_cache()
        return loss
    
    def predict(self, anchor, target, tokenizer):
        """
        Predict the similarity score between anchor and target tokens
        """
        anchor_tokens = tokenizer(anchor, padding=True, truncation=True, return_tensors='pt')
        target_tokens = tokenizer(target, padding=True, truncation=True, return_tensors='pt')


        anchor_input_ids, anchor_attention_mask = anchor_tokens['input_ids'], anchor_tokens['attention_mask']
        target_input_ids, target_attention_mask = target_tokens['input_ids'], target_tokens['attention_mask']
        
        anchor_embeddings = self.model(anchor_input_ids, anchor_attention_mask)
        target_embeddings = self.model(target_input_ids, target_attention_mask)
        return cosine_similarity(anchor_embeddings, target_embeddings)

    def on_epoch_end(self):
        """
        On epoch end
        """
        torch.cuda.empty_cache()
        if self.prediction_example is not None and self.tokenizer is not None:
            predictions = []
            anchors = []
            targets = []
            labels = []
            for index, row in self.prediction_example.iterrows():
                predictions.append(self.predict(row['anchor'], row['target'], self.tokenizer))
                anchors.append(row['anchor'])
                targets.append(row['target'])
                labels.append(row['label'])
            
            df = pd.DataFrame({'anchor': anchors, 'target': targets, 'label': labels, 'prediction': predictions})
            df.to_csv(f'predictions_{self.epoch_num}.csv', index=False)
            self.epoch_num += 1
            print(df)


    def configure_optimizers(self):
        """
        Configure the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def save_backbone(self, save_dir):
        """
        Save backbone embedding model
        """
        torch.save(self.model.state_dict(), save_dir)

class MaskedLanguageModelingModel(pl.LightningModule):
    """
    Masked language modeling model
    """
    def __init__(self, embedding_backbone, learning_rate=1e-3, loss=torch.nn.CrossEntropyLoss()):
        """
        Model name can be any of the models from the transformers library
        Freeze is a boolean to freeze the weights of the model
        Pool outputs is a boolean to pool the outputs of the model
        Learning rate is the learning rate for the optimizer
        """
        super().__init__()
        self.model = embedding_backbone
        self.fc = torch.nn.Linear(self.model.config.hidden_size,
                                  self.model.config.vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.loss = loss
        self.learning_rate = learning_rate

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model
        """
        embeddings = self.model(input_ids, attention_mask)
        pred = self.fc(embeddings)
        return self.softmax(pred)

    def training_step(self, batch, batch_idx):
        """
        Training step of the model
        """
        input_ids, attention_mask, labels = batch
        model_output = self(input_ids, attention_mask)
        loss = sel.loss(model_output.view(-1, self.config.vocab_size), labels.view(-1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model
        """
        input_ids, attention_mask, labels = batch
        model_output = self.model(input_ids, attention_mask)
        loss = self.loss(model_output, labels)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def save_backbone(self, save_dir):
        """
        Save backbone embedding model
        """
        torch.save(self.model.state_dict(), save_dir)