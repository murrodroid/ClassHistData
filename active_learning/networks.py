import torch
import torch.nn as nn

class FFNBlock(nn.Module):
    def __init__(self, dim, expansion=2, dropout=0.5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop2 = nn.Dropout(dropout)
    def forward(self, x):
        y = self.norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop2(y)
        return x + y

class network_hash_widedeep(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=512, num_blocks=2, expansion=2, dropout=0.5, use_log1p=True):
        super().__init__()
        self.use_log1p = use_log1p
        self.wide = nn.Linear(input_dim, num_classes)
        self.input = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([FFNBlock(hidden, expansion=expansion, dropout=dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, num_classes))
    def forward(self, x):
        if self.use_log1p:
            x = torch.log1p(x)
        wide = self.wide(x)
        h = self.input(x)
        for b in self.blocks:
            h = b(h)
        deep = self.head(h)
        return wide + deep