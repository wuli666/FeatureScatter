import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self,num_classes=3):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(3072, 100, bias=True)
        self.hidden2 = nn.Linear(100, 100)
        self.hidden3 = nn.Linear(100, 100)
        self.predict = nn.Linear(100, num_classes)

    def forward(self, x):
        x = torch.flatten(x,1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        output = self.predict(x)
        # output = output.view(-1)
        # output = output.view(output.size(0), 3)

        return output
