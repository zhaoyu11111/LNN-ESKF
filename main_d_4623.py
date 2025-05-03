import joblib
import numpy as np
import pandas as pd
import torch
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import math

from eskf_liquid import ESKF_LiquidNN
from test import LiquidNeuralNetwork

# Set device
device = torch.device('cpu')  # Ensure using CPU

def load_data(file_path):
    """Load data and handle missing values"""
    try:
        df = pd.read_csv(file_path)
        df.interpolate(method='linear', inplace=True)
        df.ffill(inplace=True)
        return df
    except Exception as e:
        print(f"Data loading failed: {e}")
        exit(1)

if __name__ == "__main__":
    # Configuration parameters
    CONFIG = {
        'data_path': r'D:\液态神经网络+卡尔曼滤波\dataset\4623_data.csv',
        'anchor_path': r'D:\液态神经网络+卡尔曼滤波\dataset\anchor_const4.npz',
        'model_path': r'D:\液态神经网络+卡尔曼滤波\LNN_ESKF\model\best_liquid_model.pth',  # Update model path
        'preprocessor_path': r'D:\液态神经网络+卡尔曼滤波\LNN_ESKF\model\preprocessor.pkl',
        'scaler_y_path': r'D:\液态神经网络+卡尔曼滤波\LNN_ESKF\model\scaler_y.pkl',
        'X0': [1.507, 0.016, 0.079, 0, 0, 0],
        'P0': np.diag([0.1 ** 2] * 6 + [np.deg2rad(5) ** 2] * 3)
    }

    # Load data
    df = load_data(CONFIG['data_path'])
    gt_data = df[['t_tdoa', 'pose_x', 'pose_y', 'pose_z', 'pose_qx', 'pose_qy', 'pose_qz', 'pose_qw']].values
    gt_pos = gt_data[:, 1:4]

    # Load liquid neural network model
    try:
        checkpoint = torch.load(CONFIG['model_path'], map_location=device)  # Remove weights_only=True

        # Rename keys, remove "ltc_cell." prefix
        new_state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            new_key = key.replace('ltc_cell.', '')
            new_state_dict[new_key] = value

        liquid_model = LiquidNeuralNetwork(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            output_dim=checkpoint['output_dim']
        )
        liquid_model.load_state_dict(new_state_dict)
        liquid_model.to(device)  # Move model to specified device
        liquid_model.eval()
        preprocessor = joblib.load(CONFIG['preprocessor_path'])
        scaler_y = joblib.load(CONFIG['scaler_y_path'])
    except Exception as e:
        print(f"Model loading failed: {e}")
        exit(1)

    # Load anchor coordinates
    try:
        anchor_data = np.load(CONFIG['anchor_path'])
        anchors = anchor_data['an_pos']
    except Exception as e:
        print(f"Anchor coordinate loading failed: {e}")
        exit(1)

    # Create ESKF-LiquidNN fusion instance
    K = len(df)
    eskf = ESKF_LiquidNN(
        X0=torch.tensor(CONFIG['X0'], dtype=torch.float32).to(device),
        q0=Quaternion([1, 0, 0, 0]),
        P0=torch.tensor(CONFIG['P0'], dtype=torch.float32).to(device),
        K=K,
        liquid_model=liquid_model,
        preprocessor=preprocessor,
        scaler_y=scaler_y,
        time_steps=10
    )

    timestamps = df['t_tdoa'].values

    # Main processing loop
    for k in tqdm(range(1, len(timestamps))):
        try:
            current_data = {
                't_tdoa': timestamps[k],
                'idA': df['idA'].iloc[k],
                'idB': df['idB'].iloc[k],
                'tdoa_meas': df['tdoa_meas'].iloc[k],
                'acc_x': df['acc_x'].iloc[k],
                'acc_y': df['acc_y'].iloc[k],
                'acc_z': df['acc_z'].iloc[k],
                'gyro_x': df['gyro_x'].iloc[k],
                'gyro_y': df['gyro_y'].iloc[k],
                'gyro_z': df['gyro_z'].iloc[k]
            }
            dt = timestamps[k] - timestamps[k - 1]

            # Convert current data to tensor and move to specified device
            imu_data = torch.tensor(df[['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']].iloc[k].values, dtype=torch.float32).to(device)

            # ESKF prediction step
            eskf.predict(
                imu=imu_data,
                dt=dt,
                imu_check=True,
                k=k
            )

            # LiquidNN correction step
            eskf.liquid_correct(current_data, k)
        except Exception as e:
            print(f"Step {k} processing exception: {e}, Tensor device: {imu_data.device if 'imu_data' in locals() else 'undefined'}")
            continue

    # Ensure all tensors are moved to CPU before conversion
    if isinstance(eskf.Xpo, torch.Tensor):
        est_pos = eskf.Xpo[:, :3].cpu().numpy()
    elif isinstance(eskf.Xpo, np.ndarray):
        est_pos = eskf.Xpo[:, :3]
    else:
        raise ValueError("Xpo is neither a PyTorch tensor nor a NumPy array")

    # Add estimated positions to DataFrame
    df['x_estimate'] = est_pos[:, 0]
    df['y_estimate'] = est_pos[:, 1]
    df['z_estimate'] = est_pos[:, 2]

    # Save the DataFrame with estimates to a new CSV file
    output_csv_file = "4623_LNN_ESKF.csv"  # New file name
    df.to_csv(output_csv_file, index=False)
    print(f"\nCSV file with estimates saved as: {output_csv_file}")

    # Error analysis
    x_error = df['pose_x'] - df['x_estimate']
    y_error = df['pose_y'] - df['y_estimate']
    z_error = df['pose_z'] - df['z_estimate']

    abs_x_error = np.abs(x_error)
    abs_y_error = np.abs(y_error)
    abs_z_error = np.abs(z_error)

    mean_abs_x = np.mean(abs_x_error)
    mean_abs_y = np.mean(abs_y_error)
    mean_abs_z = np.mean(abs_z_error)
    mean_abs_total = np.mean(np.sqrt(abs_x_error**2 + abs_y_error**2 + abs_z_error**2))

    rms_x = math.sqrt(np.mean(x_error ** 2))
    rms_y = math.sqrt(np.mean(y_error ** 2))
    rms_z = math.sqrt(np.mean(z_error ** 2))
    rms_total = math.sqrt(rms_x ** 2 + rms_y ** 2 + rms_z ** 2)

    print(f"\nError Analysis:")
    print(f"X-axis Mean Absolute Error (MAE): {mean_abs_x:.4f} m")
    print(f"Y-axis Mean Absolute Error (MAE): {mean_abs_y:.4f} m")
    print(f"Z-axis Mean Absolute Error (MAE): {mean_abs_z:.4f} m")
    print(f"Total Mean Absolute Error (MAE): {mean_abs_total:.4f} m")
    print(f"X-axis Root Mean Square Error (RMSE): {rms_x:.4f} m")
    print(f"Y-axis Root Mean Square Error (RMSE): {rms_y:.4f} m")
    print(f"Z-axis Root Mean Square Error (RMSE): {rms_z:.4f} m")
    print(f"Total Root Mean Square Error (RMSE): {rms_total:.4f} m")

    # Visualization
    # 可视化
    plt.figure(figsize=(8, 7))

    # 3D轨迹对比
    fig = plt.figure(figsize=(8, 7))

    ax = fig.add_subplot(111, projection='3d')
    ax.plot(df['pose_x'], df['pose_y'], df['pose_z'], color='b', linewidth=2.0, alpha=0.7, label='Ground Truth')
    ax.plot(df['x_estimate'], df['y_estimate'], df['z_estimate'], color='g', linewidth=3.0, alpha=0.5, label='Estimate')
    ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], color='Teal', s=100, alpha=0.5, label='Anchors')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')

    ax.legend()
    ax.grid(True)

    # Save vector graphic
    output_vector_graphic_file = "4623_LNN_eskf.png"
    plt.savefig(output_vector_graphic_file, format='png', dpi=300)  # Save with high resolution
    print(f"\nVector graphic saved as: {output_vector_graphic_file}")

    plt.tight_layout()
    plt.show()
