import torch
import torch.nn as nn

class WeightedMAELoss(nn.Module):
    def __init__(self, num_outputs):
        super(WeightedMAELoss, self).__init__()
        self.num_outputs = num_outputs
        self.weights = torch.ones(num_outputs, requires_grad=False)

    def forward(self, predictions, targets):
        # dynamic update based on error
        errors = torch.abs(predictions - targets)
        self.weights = errors / errors.sum()
        losses = self.weights * errors
        return losses.mean()

