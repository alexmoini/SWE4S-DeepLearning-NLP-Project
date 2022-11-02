import torch
import torch.nn.functional as F

def cosine_similarity(anchor_embeddings, target_embeddings):
    """
    Cosine similarity
    """
    return torch.cosine_similarity(anchor_embeddings, target_embeddings, dim=1)

class CosineEmbeddingLoss:
    """
    Cosine embedding loss
    """
    def __init__(self, regression_loss_function=F.mse_loss):
        """
        Regression loss function is the loss function to use for regression
        """
        self.regression_loss_function = regression_loss_function

    def __call__(self, anchor_embeddings, target_embeddings, labels):
        """
        Call the loss function
        """
        return self.regression_loss_function(cosine_similarity(anchor_embeddings, target_embeddings), labels)


