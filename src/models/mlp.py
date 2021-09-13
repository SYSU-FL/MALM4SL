import torch.nn as nn
import torch.nn.functional as F

## Define the NN architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class MLP_Extractor(nn.Module):
    def __init__(self):
        super(MLP_Extractor, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
    def forward(self, x):
        x = x.view(-1, 28*28)
        return F.relu(self.fc1(x))

class MLP_Classifier(nn.Module):
    def __init__(self):
        super(MLP_Classifier, self).__init__()
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))