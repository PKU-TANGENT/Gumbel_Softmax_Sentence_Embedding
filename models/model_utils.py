import torch.nn as nn
class CosSimilarityWithTemp(nn.Module):
    """
    Cosine similarity with temperature
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp