from lightai.core import *

class MAP:
    def __init__(self):
        self.scores = []

    def __call__(self, predict, target):
        """
        :param predict: a batch of predict
        :param target: a batch of target
        :return: None
        """
        target = target.view(-1, 1)
        _, idx = predict.topk(3)
        score = (idx == target).float() @ torch.tensor([1, 0.5, 1/3], device='cuda').view(-1, 1)
        self.scores.append(score)

    def res(self):
        res = torch.cat(self.scores).mean()
        self.scores = []
        return res