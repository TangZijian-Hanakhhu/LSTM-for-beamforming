import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from downlink_beamforming_LSTM import *

def generate_complex_downlink_channel(batch_size, S, M, K, Lp, delta_number, dl_cor, rician_factor, pilot_length, user_mobility=False, doppler_shift=0.01):
    h_dl_K = np.zeros([batch_size, S, M, K], dtype=np.complex64)

    for k in range(K):
        theta = 60.0 * np.random.rand(batch_size, Lp) - 30.0
        alphaX = np.sqrt(0.5) * (np.random.standard_normal([batch_size, Lp]) + 1j * np.random.standard_normal([batch_size, Lp]))
        alpha1 = np.sqrt(0.5) * (np.random.standard_normal([batch_size, S, Lp]) + 1j * np.random.standard_normal([batch_size, S, Lp]))
        alpha1[:, 0, :] = alphaX

        for s in range(1, S):
            noise = np.sqrt(0.5) * (np.random.standard_normal([batch_size, Lp]) + 1j * np.random.standard_normal([batch_size, Lp]))
            alpha1[:, s, :] = alpha1[:, s - 1, :] * dl_cor + noise * np.sqrt(1 - np.square(dl_cor))

        h_dl = np.zeros([batch_size, S, M], dtype=np.complex64)

        for p in range(Lp):
            for m in range(M):
                for s in range(S):
                    phase_shift = np.exp(1j * 2 * np.pi * delta_number * m * np.sin(theta[:, p] / 180 * np.pi))
                    if user_mobility:
                        phase_shift *= np.exp(1j * 2 * np.pi * doppler_shift * s)
                    if rician_factor > 0:
                        direct_path = np.sqrt(rician_factor / (1 + rician_factor))
                        scatter_path = np.sqrt(1 / (1 + rician_factor)) * alpha1[:, s, p]
                        gain = (1.0 / np.sqrt(Lp)) * (direct_path + scatter_path) * phase_shift
                    else:
                        gain = (1.0 / np.sqrt(Lp)) * alpha1[:, s, p] * phase_shift
                    h_dl[:, s, m] += gain

        # 引入导频长度进行信道估计模拟
        estimated_h_dl = np.zeros_like(h_dl)
        for s in range(S):
            if s < pilot_length:
                estimated_h_dl[:, s, :] = h_dl[:, s, :]
            else:
                estimated_h_dl[:, s, :] = np.mean(h_dl[:, :pilot_length, :], axis=1)

        h_dl_K[:, :, :, k] = estimated_h_dl
    return h_dl_K
M = 64  # 天线数量
K = 2   # 用户数量
S = 8   # 时间步长
Lp = 5  # 多径数量
batch_size = 1024  # 批次大小
Power = 1.0
hidden_size = K * M * 2
input_size = K * M * 2
SNR_db = 15
SNR = 10 ** (SNR_db / 10)
sigma2 = Power / SNR
epochs = 4000
delta = 100
delta_number = 0.05 * (delta * 1e6 + 3e9) / 3e8
dl_cor = 0.9998  # 多普勒相关性
rician_factor = 0  # 莱斯因子，0为瑞利衰落
Bandwidth = 20  # 这里带宽是指量化比特数
N0 = sigma2 # 噪声
pilot_length = 8
user_mobility = True
multi_cell_interference = True  # 多小区干扰
doppler_shift = 0.01  # 多普勒频移
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DownlinkLSTM(M, K, hidden_size, batch_size, Power, Bandwidth, N0).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.9)
h_dl_ture = torch.from_numpy(generate_complex_downlink_channel(batch_size, S, M, K, Lp, delta_number, dl_cor, rician_factor, pilot_length=pilot_length, user_mobility=user_mobility, doppler_shift=doppler_shift)).to(torch.complex64).to(device)
h_dl_real = torch.cat([h_dl_ture.real, h_dl_ture.imag], dim=2)
h_dl_real = h_dl_real.view(batch_size, S, -1)

best_loss = float('inf')
best_model_path = 'best_model.pth'

# 训练
losses = []
for epoch in range(epochs):
    model.train()
    v_pred = model(h_dl_real)
    loss = model.rate_loss(h_dl_ture, v_pred, M, K, sigma2)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    losses.append(loss.item())
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save({'model_state_dict': model.state_dict(), 'losses': losses}, best_model_path)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

print(f'The best model was saved to {best_model_path}')