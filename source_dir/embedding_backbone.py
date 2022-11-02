import torch
import transformers


class TextEmbeddingBackbone(torch.nn.Module):
    """
    Embedding backbone for down stream tasks
    """
    def __init__(self, model_architecture, checkpoint_dir=None, freeze=False, pool_outputs=True):
        """
        Model name can be any of the models from the transformers library
        Freeze is a boolean to freeze the weights of the model
        Pool outputs is a boolean to pool the outputs of the model
        """
        super().__init__()
        # loop through state_dict and replace all the keys with the new ones
        self.model = transformers.AutoModel.from_pretrained(model_architecture)
        if checkpoint_dir is not None:
            checkpoint = torch.load(checkpoint_dir)
            for key in list(checkpoint.keys()):
                new_key = key.replace('model.', '')
                checkpoint[new_key] = checkpoint.pop(key)
            self.model.load_state_dict(checkpoint)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.pool_outputs = pool_outputs
        # for ease of use in tuners
        self.config = self.model.config

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model
        """
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if self.pool_outputs:
            return self.__mean_pooling(model_output, attention_mask)
        else:
            return model_output

    def __mean_pooling(self, model_output, attention_mask):
        """
        mean pooling of the model output, pools along the sequence dimension, all the padded tokens are ignored
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
