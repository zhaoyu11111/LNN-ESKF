import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from numpy.lib.stride_tricks import sliding_window_view


class LiquidNeuralNetwork(nn.Module):
    """液态神经网络模型（需与训练时结构完全一致）"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 液态时间常数单元
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.U = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.nonlinearity = nn.Tanh()

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x形状: (batch_size, time_steps, input_dim)
        batch_size, time_steps, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        for t in range(time_steps):
            current_input = x[:, t, :]
            dh = (-h + self.nonlinearity(
                torch.mm(h, self.W) +
                torch.mm(current_input, self.U) +
                self.b
            )) / self.tau.clamp(min=0.01)
            h = h + 0.1 * dh

        return self.fc(h)


def load_model_and_predict(new_data_path, time_steps=10, batch_size=512):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载预处理组件
    preprocessor = joblib.load('model/preprocessor.pkl')
    scaler_y = joblib.load('model/scaler_y.pkl')

    # 加载模型参数
    checkpoint = torch.load('model/best_liquid_model.pth', map_location=device)
    model = LiquidNeuralNetwork(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载新数据
    new_data = pd.read_csv(new_data_path)

    # 数据预处理
    def create_dataset(X, time_steps):
        # 使用滑动窗口视图生成窗口数据
        return sliding_window_view(X, window_shape=(time_steps, X.shape[1])).squeeze()

    # 提取特征列
    feature_columns = ['t_tdoa', 'idA', 'idB', 'tdoa_meas',
                       'acc_x', 'acc_y', 'acc_z',
                       'gyro_x', 'gyro_y', 'gyro_z']
    X_new = new_data[feature_columns].values

    # 标准化
    X_processed = preprocessor.transform(X_new)

    # 应用滑动窗口
    X_windowed = create_dataset(X_processed, time_steps)

    # 转换为张量
    X_tensor = torch.tensor(X_windowed, dtype=torch.float32).to(device)

    # 分批推理
    predictions = []
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size]
        with torch.no_grad():
            pred_scaled = model(batch).cpu().numpy()
        predictions.append(pred_scaled)
    pred_scaled = np.concatenate(predictions, axis=0)

    # 逆标准化
    pred_original = scaler_y.inverse_transform(pred_scaled)

    # 获取对应时间戳
    timestamps = new_data['t_tdoa'].values[time_steps:]

    return timestamps, pred_original


if __name__ == "__main__":
    # 使用示例
    test_data_path = 'dataset/test_data.csv'  # 替换为实际测试数据路径

    # 执行预测
    timestamps, predictions = load_model_and_predict(test_data_path, time_steps=10)

    # 打印前5个预测结果
    print("\n预测结果示例：")
    for i in range(5):
        print(f"时间: {timestamps[i]:.2f}s | 位置: {predictions[i, :3]} | 四元数: {predictions[i, 3:]}")

    # 保存结果
    result_df = pd.DataFrame(predictions,
                             columns=['pose_x', 'pose_y', 'pose_z',
                                      'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw'])
    result_df['t_tdoa'] = timestamps
    result_df.to_csv('liquid_predictions.csv', index=False)
    print("\n预测结果已保存至 liquid_predictions.csv")