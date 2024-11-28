import torch
import torch.nn as nn
import numpy as np
class DownlinkLSTM(nn.Module):
    def __init__(self, M, K, hidden_size, batch_size, Power, Bandwidth, N0):
        super(DownlinkLSTM, self).__init__()
        self.M = M
        self.K = K
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.Power = Power
        self.Bandwidth = Bandwidth
        self.N0 = N0
        self.lstm = nn.LSTM(input_size=M * K * 2, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, M * K * 2)
        self.dropout = nn.Dropout(0.5)
        self.qual = nn.Linear(256, Bandwidth)
        self.dfc1 = nn.Linear(Bandwidth, K * M * 2)
        self.dfc2 = nn.Linear(K * M * 2, K * M * 2)

    def forward(self, h_dl):
        h_0 = torch.zeros(1, h_dl.size(0), self.hidden_size).to(h_dl.device)
        c_0 = torch.zeros(1, h_dl.size(0), self.hidden_size).to(h_dl.device)
        lstm_out, _ = self.lstm(h_dl, (h_0, c_0))
        lstm_out = self.dropout(lstm_out)
        v_pred = self.fc(lstm_out[:, -1, :])
        norm_v = torch.sqrt(torch.tensor(self.Power)) / torch.norm(v_pred, dim=1, keepdim=True)
        v_norm = v_pred * norm_v
        qual_output = torch.relu(self.qual(v_norm))
        v_mod = torch.tanh(self.dfc2(self.dfc1(qual_output)))
        return v_mod

    def rate_loss(self, h_dl, v, M, K, sigma2):
        V_complex = torch.complex(v[:, :M * K], v[:, M * K:])
        v_reshape = V_complex.view(-1, M, K)
        loss = 0
        noise_power = self.N0 * self.Bandwidth
        for k in range(K):
            h_k = h_dl[:, :, :, k]
            signal_power = torch.abs(torch.sum(h_k * v_reshape.unsqueeze(1)[:, :, :, k], dim=2)) ** 2
            interference = noise_power + torch.sum(torch.abs(torch.matmul(h_k, v_reshape)) ** 2, dim=2).squeeze(1) - signal_power
            rate = torch.log2(1 + (signal_power) / interference).mean()
            loss -= rate
        return loss