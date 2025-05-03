import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os


# 新增：定义模型保存目录
MODEL_DIR = 'model'
os.makedirs(MODEL_DIR, exist_ok=True)  # 创建目录（如果不存在）

base_path = r'D:\液态神经网络+卡尔曼滤波\dataset'
file_paths = [

    os.path.join(base_path, '4121_data.csv'),
    os.path.join(base_path, '4122_data.csv'),
    os.path.join(base_path, '4123_data.csv'),
    os.path.join(base_path, '4221_data.csv'),
    os.path.join(base_path, '4222_data.csv'),
    os.path.join(base_path, '4223_data.csv'),
    os.path.join(base_path, '4321_data.csv'),
    os.path.join(base_path, '4322_data.csv'),
    os.path.join(base_path, '4323_data.csv'),


]

# 读取多个CSV文件
print("正在加载数据...")
data_frames = []
for path in file_paths:
    df = pd.read_csv(path)
    data_frames.append(df)
data = pd.concat(data_frames, ignore_index=True)
print(f"数据加载完成，共加载{len(file_paths)}个文件，总样本数：{len(data)}")

# 定义输入特征和标签的列名
input_columns = ['t_tdoa', 'idA', 'idB', 'tdoa_meas',
                 'acc_x', 'acc_y', 'acc_z',
                 'gyro_x', 'gyro_y', 'gyro_z']
label_columns = ['pose_x', 'pose_y', 'pose_z',
                 'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw']

# 分离特征和标签
X = data[input_columns]
y = data[label_columns].values

# 定义预处理管道
numeric_features = ['t_tdoa', 'tdoa_meas',
                    'acc_x', 'acc_y', 'acc_z',
                    'gyro_x', 'gyro_y', 'gyro_z']
categorical_features = ['idA', 'idB']

preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), numeric_features),
                  ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                  ])

# 应用特征预处理
X_processed = preprocessor.fit_transform(X)

# 应用标签标准化
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# 保存预处理管道和标准化器
# 修改预处理器的保存路径
joblib.dump(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.pkl'))
joblib.dump(scaler_y, os.path.join(MODEL_DIR, 'scaler_y.pkl'))
print("预处理管道和标准化器已保存。")



# 创建时间窗口数据集
def create_dataset(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        window_X = X[i:(i + time_steps)]
        window_y = y[i + time_steps]
        Xs.append(window_X)
        ys.append(window_y)
    return np.array(Xs), np.array(ys)


time_steps = 10
X_final, y_final = create_dataset(X_processed, y_scaled, time_steps=time_steps)

# 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
batch_size = 256
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 液态神经网络实现
class LTCCell(nn.Module):
    """液态时间常数网络单元"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 可学习参数
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.U = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.b = nn.Parameter(torch.zeros(hidden_dim))

        # 时间常数参数
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.nonlinearity = nn.Tanh()

    def forward(self, t, h):
        dhdt = (-h + self.nonlinearity(torch.mm(h, self.W) +
                                       torch.mm(self.current_input, self.U) +
                                       self.b)) / self.tau.clamp(min=0.01)
        return dhdt


class LiquidNeuralNetwork(nn.Module):
    """液态神经网络模型"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ltc_cell = LTCCell(input_dim, hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x形状: (batch_size, time_steps, input_dim)
        batch_size, time_steps, _ = x.size()

        # 初始化隐藏状态
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        # 迭代处理每个时间步
        for t in range(time_steps):
            # 设置当前输入
            self.ltc_cell.current_input = x[:, t, :]

            # 使用欧拉方法进行微分方程求解
            h = h + 0.1 * self.ltc_cell(0, h)  # 固定时间步长0.1

        # 最终全连接层
        out = self.fc(h)
        return out


# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的设备: {device}")

input_dim = X_train.shape[2]
hidden_dim = 128
output_dim = y_train.shape[1]

model = LiquidNeuralNetwork(input_dim=input_dim,
                            hidden_dim=hidden_dim,
                            output_dim=output_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)

# 训练参数
num_epochs = 150
best_test_loss = float('inf')

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{num_epochs}") as tepoch:
        for inputs, labels in tepoch:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            tepoch.set_postfix(loss=loss.item())

    # 验证阶段
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # 保存最佳模型
    best_model_path = os.path.join(MODEL_DIR, 'best_liquid_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim
    }, best_model_path)
    print("发现新的最佳模型，已保存！")

# 测试模型
model.load_state_dict(torch.load(best_model_path)['model_state_dict'])
model.eval()

test_loss = 0.0
predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        predictions.append(outputs.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
print(f"\n最终测试损失: {avg_test_loss:.4f}")

# 保存预测结果
predictions = np.concatenate(predictions, axis=0)
predictions_rescaled = scaler_y.inverse_transform(predictions)
np.savetxt("liquid_predictions.csv", predictions_rescaled, delimiter=",")
print("预测结果已保存。")