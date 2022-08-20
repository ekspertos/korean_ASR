import logging

import torch
import torch.nn as nn

from typing import Any
from typing import List


# filtering linear regression model
class FilteringLinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.u = nn.Parameter(torch.Tensor((1)))
        self.beta = nn.Parameter(torch.Tensor(1))
        self.reset_parameter()

    def reset_parameter(self):
        nn.init.constant_(self.u, 0.0)
        nn.init.constant_(self.beta, 0.0)

    def forward(self, fuse_score, length):
        sqrt_length = torch.sqrt(length)
        filtering_score = (fuse_score - self.u * length - self.beta) / sqrt_length
        return filtering_score


# train linear regression model
def train(
        model,
        log_score: List[float],
        script: List[str],
        learning_rate: float = 1e-3,
        epoch: int = 5
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    script_length = torch.tensor([len(s) for s in script]).to(device)
    log_score = torch.tensor(log_score).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    for idx in range(epoch):
        optimizer.zero_grad()
        filtering_score = model(log_score, script_length)
        loss = loss_fn(filtering_score, torch.zeros_like(filtering_score))
        loss.backward()
        optimizer.step()
        loss_value = loss.detach().cpu().item()
        if idx % 1000:
            print(f"epoch:{idx+1}, loss:{loss_value}")
    model = model.to("cpu")
    return model


def validate(
        model,
        log_score: List[float],
        script: List[str],
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    script_length = torch.tensor([len(s) for s in script]).to(device)
    log_score = torch.tensor(log_score).to(device)
    model.eval()
    with torch.no_grad():
        score = model(log_score, script_length).cpu().numpy()

    return score
