'''PyTorch-compatible losses and loss functions.
'''

import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    '''Cross-entropy.

    See: http://cs231n.github.io/neural-networks-2/#losses
    '''

    def __init__(self, weight=None, size_average=True, ignore_index=255):
        '''Creates an `CrossEntropyLoss2d` instance.

        Args:
          weight: rescaling weight for each class.
          size_average: if true average losses for minibatch; otherwise sum up losses for minibatch.
          ignore_index: input value that is ignored.
        '''

        super().__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(nn.functional.log_softmax(inputs, dim=1), targets)


class FocalLoss2d(nn.Module):
    '''Focal Loss.

    Reduces loss for well-classified samples putting focus on hard mis-classified samples.

    See: https://arxiv.org/abs/1708.02002
    '''

    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        '''Creates a `FocalLoss2d` instance.

        Args:
          gamma: the focusing parameter; if zero this loss is equivalent with `CrossEntropyLoss2d`.
          weight: rescaling weight for each class.
          size_average: if true average losses for minibatch; otherwise sum up losses for minibatch.
          ignore_index: input value that is ignored.
        '''

        super().__init__()

        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)
        self.gamma = gamma

    def forward(self, inputs, targets):
        penalty = (1 - nn.functional.softmax(inputs, dim=1)) ** self.gamma
        return self.nll_loss(penalty * nn.functional.log_softmax(inputs, dim=1), targets)
