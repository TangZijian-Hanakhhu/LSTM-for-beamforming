import matplotlib.pyplot as plt
import torch

# 假设训练过程中损失值被记录在一个列表中，例如：
epochs = 4000
losses = []

# 从模型文件中读取损失值和保存点信息（假设在训练过程中，我们保存了模型状态和损失值）
checkpoint = torch.load('best_model.pth')
losses = checkpoint.get('losses', [])  # 从保存的模型文件中获取损失值

# 绘制损失值的折线图
def plot_loss_curve(losses, epochs):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, label='Training Loss', color='b', linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 调用函数绘制损失值曲线
plot_loss_curve(losses, len(losses))

# 可选：保存图像
# plt.savefig('training_loss_curve.png')

# 附加：训练代码中可以直接将损失记录在模型检查点中，例如：
# torch.save({'model_state_dict': model.state_dict(), 'losses': losses}, 'best_model.pth')