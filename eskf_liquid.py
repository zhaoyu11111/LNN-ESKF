# eskf_liquid.py (Modified ESKF_LiquidNN Class ONLY)

import numpy as np
import pandas as pd
from scipy import linalg
import math
from pyquaternion import Quaternion
import torch
from torch import nn

# Assuming StandardScaler and OneHotEncoder might be needed for type hints or inspection
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Constants ---
w_accxyz = 2.0
w_gyro_rpy = 0.1
GRAVITY_MAGNITUDE = 9.81
DEG_TO_RAD = math.pi / 180.0
e3 = np.array([0, 0, 1]).reshape(-1, 1)

_TRAIN_NUMERIC_FEATURES = ['t_tdoa', 'tdoa_meas', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
_TRAIN_CATEGORICAL_FEATURES = ['idA', 'idB']
_RAW_FEATURE_COLUMNS = _TRAIN_NUMERIC_FEATURES + _TRAIN_CATEGORICAL_FEATURES
_NUM_RAW_FEATURES = len(_RAW_FEATURE_COLUMNS) # Should be 10

class ESKF_LiquidNN:
    def __init__(self, X0, q0, P0, K, liquid_model, preprocessor, scaler_y, time_steps=10):
        # --- State Storage (using NumPy) ---
        self.f = np.zeros((K, 3))
        self.omega = np.zeros((K, 3))
        self.q_list = np.zeros((K, 4))
        self.R_list = np.zeros((K, 3, 3))
        self.Xpr = np.zeros((K, 6))
        self.Xpo = np.zeros((K, 6))
        self.Ppr = np.zeros((K, 9, 9))
        self.Ppo = np.zeros((K, 9, 9))

        # --- Initialize State ---
        X0_np = X0.cpu().numpy() if isinstance(X0, torch.Tensor) else np.asarray(X0)
        P0_np = P0.cpu().numpy() if isinstance(P0, torch.Tensor) else np.asarray(P0)

        if X0_np.shape != (6,) and X0_np.shape != (6, 1):
             raise ValueError(f"Initial state X0 has unexpected shape: {X0_np.shape}")
        if P0_np.shape != (9, 9):
             raise ValueError(f"Initial covariance P0 has unexpected shape: {P0_np.shape}")

        self.Xpr[0] = X0_np.flatten()
        self.Xpo[0] = X0_np.flatten()
        self.Ppr[0] = P0_np
        self.Ppo[0] = P0_np
        self.q_list[0, :] = q0.elements
        self.R_list[0] = q0.rotation_matrix
        self.R = q0.rotation_matrix

        self.Fi = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3))],
            [np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.eye(3)]
        ])

        # --- LNN Parameters ---
        self.liquid_model = liquid_model
        self.scaler_y = scaler_y
        self.time_steps = time_steps
        self.input_buffer = []
        self.device = torch.device('cpu')
        self.liquid_model.to(self.device)
        self.liquid_model.eval()

        # --- Pre-extract transformers and indices ---
        try:
            self.scaler_num = preprocessor.named_transformers_['num']
            self.encoder_cat = preprocessor.named_transformers_['cat']

            # Check consistency of feature lists
            if set(_TRAIN_NUMERIC_FEATURES + _TRAIN_CATEGORICAL_FEATURES) != set(_RAW_FEATURE_COLUMNS):
                raise ValueError("Internal feature list definitions are inconsistent.")

            self.num_indices = [ _RAW_FEATURE_COLUMNS.index(f) for f in _TRAIN_NUMERIC_FEATURES ]
            self.cat_indices = [ _RAW_FEATURE_COLUMNS.index(f) for f in _TRAIN_CATEGORICAL_FEATURES ]

            # Ensure indices cover all expected raw features without duplicates and are within bounds
            all_indices = sorted(self.num_indices + self.cat_indices)
            if len(all_indices) != _NUM_RAW_FEATURES or all_indices != list(range(_NUM_RAW_FEATURES)):
                 raise ValueError(f"Generated indices {all_indices} do not match expected range [0...{_NUM_RAW_FEATURES-1}] based on _RAW_FEATURE_COLUMNS.")

            # Determine expected dimension after preprocessing
            n_num_features = len(self.num_indices)
            try: # Handle potential issues if categories_ is not as expected
                n_cat_onehot_features = sum(len(cats) for cats in self.encoder_cat.categories_)
            except AttributeError:
                 raise ValueError("Could not determine one-hot feature count from encoder_cat.categories_")
            self.expected_processed_dim = n_num_features + n_cat_onehot_features

            # Get expected input dim for LNN model
            if hasattr(self.liquid_model, 'ltc_cell') and hasattr(self.liquid_model.ltc_cell, 'U'):
                 self.model_input_dim = self.liquid_model.ltc_cell.U.shape[0]
            else:
                 print("Warning: Could not directly verify LNN model input dimension from model structure. Assuming it matches preprocessor output.")
                 self.model_input_dim = self.expected_processed_dim

            if self.expected_processed_dim != self.model_input_dim:
                 print(f"CRITICAL Warning: Dimension mismatch detected during init! Preprocessor outputs {self.expected_processed_dim} features, but model expects {self.model_input_dim}.")
                 # Consider raising an error here if this mismatch is fatal

        except KeyError:
             raise ValueError("Preprocessor object is missing 'num' or 'cat' transformer.")
        except AttributeError:
             raise ValueError("Preprocessor object does not seem to be a ColumnTransformer or scaler/encoder objects are missing expected attributes.")
        except Exception as e:
            raise RuntimeError(f"Error extracting transformers or indices during init: {e}")


    def _preprocess_window(self, raw_window_np):
        """ Applies preprocessing directly using fitted scaler and encoder objects. """
        # --- Input Shape Check ---
        # Check number of rows (time steps)
        if raw_window_np.shape[0] != self.time_steps:
            print(f"Error in _preprocess_window: Expected {self.time_steps} time steps, but got window shape {raw_window_np.shape}")
            return None
        # Check number of columns (raw features) -> THIS IS LIKELY WHERE THE ISSUE LIES
        if raw_window_np.shape[1] != _NUM_RAW_FEATURES:
            print(f"CRITICAL Error in _preprocess_window: Expected {_NUM_RAW_FEATURES} raw features (columns), but got window shape {raw_window_np.shape}")
            # Print some info to help debug the input source (main.py)
            print("This likely means the data row passed from main.py was incorrect.")
            # print("First row of problematic window:", raw_window_np[0, :]) # This might fail if only 1 col
            return None
        # --- End Input Shape Check ---

        try:
            # --- Indexing ---
            # These lines will fail if raw_window_np has only 1 column due to the check above
            numeric_data = raw_window_np[:, self.num_indices]
            categorical_data = raw_window_np[:, self.cat_indices]
            # ---------------

            # Check for NaN/Inf before scaling
            if not np.all(np.isfinite(numeric_data)):
                print("Warning: Non-finite values found in numeric data before scaling. Replacing with 0.")
                numeric_data = np.nan_to_num(numeric_data, nan=0.0, posinf=0.0, neginf=0.0)

            scaled_numeric = self.scaler_num.transform(numeric_data)
            encoded_categorical = self.encoder_cat.transform(categorical_data).toarray()

            # Combine features
            processed_window = np.concatenate((scaled_numeric, encoded_categorical), axis=1)

            # Final output shape check
            if processed_window.shape != (self.time_steps, self.expected_processed_dim):
                 print(f"Error: Final processed window shape {processed_window.shape} doesn't match expected shape {(self.time_steps, self.expected_processed_dim)}.")
                 return None

            return processed_window

        except IndexError as ie:
             # This catch might be redundant now due to the input shape check, but keep for safety
             print(f"IndexError during manual preprocessing slicing: {ie}. This should not happen if input shape check passed.")
             print(f"Raw window shape was: {raw_window_np.shape}")
             print(f"Num indices: {self.num_indices}")
             print(f"Cat indices: {self.cat_indices}")
             return None
        except ValueError as ve:
             print(f"ValueError during transform (scaler/encoder): {ve}. Check data ranges/categories.")
             return None
        except Exception as e:
            print(f"Unexpected error during manual preprocessing: {e}")
            return None

    # --- predict method remains the same as the corrected one from the previous response ---
    def predict(self, imu, dt, imu_check, k):
        # ... (Keep the version where R_f = R_km1_po.rotate(f_k) is removed) ...
        # Convert input tensor to NumPy if necessary
        if isinstance(imu, torch.Tensor):
            imu_np = imu.cpu().numpy()
        else:
            imu_np = np.asarray(imu) # Ensure it's a NumPy array

        if imu_np.shape != (6,):
            print(f"Warning: Unexpected IMU data shape {imu_np.shape} at step {k}. Expected (6,). Skipping predict step.")
            if k > 0:
                 self.Xpr[k] = self.Xpo[k-1].copy()
                 self.Ppr[k] = self.Ppo[k-1].copy()
                 self.q_list[k] = self.q_list[k-1].copy()
                 self.R_list[k] = self.R_list[k-1].copy()
            else:
                 raise ValueError("Cannot predict at step 0 without valid IMU handling")
            # Still need to set posterior = prior for skipped step if correction is also skipped
            self.Xpo[k] = self.Xpr[k].copy()
            self.Ppo[k] = self.Ppr[k].copy()
            self.R = self.R_list[k] if k > 0 else self.R # Use initial R if k=0
            return # Skip rest of predict

        # --- Process Noise Covariance Qi ---
        Vi = (w_accxyz ** 2) * (dt ** 2) * np.eye(3)
        Thetai = (w_gyro_rpy ** 2) * (dt ** 2) * np.eye(3)
        Qi = np.block([
            [Vi, np.zeros((3, 3))],
            [np.zeros((3, 3)), Thetai]
        ])

        R_km1_po = self.R

        if imu_check:
            acc_m = imu_np[:3]
            omega_m = imu_np[3:]
            f_k = acc_m * GRAVITY_MAGNITUDE
            omega_k_rad = omega_m * DEG_TO_RAD
            self.f[k] = f_k
            self.omega[k] = omega_k_rad
        else:
             if k == 0:
                 print("Warning: imu_check=False at step 0. Using zero IMU values.")
                 f_k = np.zeros(3); omega_k_rad = np.zeros(3)
             else:
                 f_k = self.f[k-1]; omega_k_rad = self.omega[k-1]
             self.f[k] = f_k; self.omega[k] = omega_k_rad

        # --- Nominal State Prediction ---
        p_km1_po = self.Xpo[k - 1, :3]
        v_km1_po = self.Xpo[k - 1, 3:6]
        q_km1_po = Quaternion(self.q_list[k - 1, :])

        term = R_km1_po.dot(f_k.reshape(-1, 1)) - GRAVITY_MAGNITUDE * e3

        p_k_pr = p_km1_po + v_km1_po * dt + 0.5 * np.squeeze(term) * dt ** 2
        v_k_pr = v_km1_po + np.squeeze(term) * dt

        dw = omega_k_rad * dt
        dqk = Quaternion(self.zeta(dw))
        q_k_pr = q_km1_po * dqk
        q_k_pr = q_k_pr.normalised

        self.Xpr[k, :3] = p_k_pr
        self.Xpr[k, 3:6] = v_k_pr
        self.q_list[k, :] = q_k_pr.elements
        self.R_list[k] = q_k_pr.rotation_matrix

        # --- Error State Covariance Prediction ---
        Fx = np.eye(9)
        Fx[0:3, 3:6] = dt * np.eye(3)
        Fx[0:3, 6:9] = -0.5 * dt**2 * R_km1_po.dot(self.cross(f_k))
        Fx[3:6, 6:9] = -dt * R_km1_po.dot(self.cross(f_k))
        delta_rot_matrix = linalg.expm(self.cross(-dw))
        Fx[6:9, 6:9] = delta_rot_matrix

        P_km1_po = self.Ppo[k - 1]
        P_k_pr = Fx @ P_km1_po @ Fx.T + self.Fi @ Qi @ self.Fi.T

        self.Ppr[k] = 0.5 * (P_k_pr + P_k_pr.T)

        # --- Default Posterior = Prior (before correction) ---
        self.Xpo[k] = self.Xpr[k].copy()
        self.Ppo[k] = self.Ppr[k].copy()
        self.R = self.R_list[k]


    # --- liquid_correct method remains the same as the corrected one from the previous response ---
    def liquid_correct(self, raw_data_row_np, k):
        """ ESKF Correction Step using Liquid NN prediction. """
        # --- Buffer Management ---
        # Check input shape before appending

        def liquid_correct(self, raw_data_row_np, k):
            """ ESKF Correction Step using Liquid NN prediction. """

            # --- 在函数入口处添加类型检查 ---
            if not isinstance(raw_data_row_np, np.ndarray):
                print(
                    f"CRITICAL ERROR inside liquid_correct at Step {k}: Received object of type {type(raw_data_row_np)}, expected np.ndarray!")
                # 打印传入的对象帮助调试
                # print(f"Received object content: {raw_data_row_np}")
                # 跳过当前步骤的处理
                self.Xpo[k] = self.Xpr[k].copy()  # 保持状态为预测值
                self.Ppo[k] = self.Ppr[k].copy()
                # R_list[k] 已经在 predict 中设置了先验值
                self.R = self.R_list[k]  # 确保 self.R 更新为当前(先验)值
                return  # 退出此函数
            # --- 类型检查结束 ---

            # Check input shape before appending (原有的检查)
            if raw_data_row_np.ndim != 1 or raw_data_row_np.shape[0] != _NUM_RAW_FEATURES:
                print(
                    f"Error in liquid_correct: raw_data_row_np at step {k} has unexpected shape {raw_data_row_np.shape}. Expected ({_NUM_RAW_FEATURES},). Skipping correction.")
                self.Xpo[k] = self.Xpr[k].copy()  # 保持状态为预测值
                self.Ppo[k] = self.Ppr[k].copy()
                self.R = self.R_list[k]
                return

        if raw_data_row_np.ndim != 1 or raw_data_row_np.shape[0] != _NUM_RAW_FEATURES:
             print(f"Error in liquid_correct: raw_data_row_np at step {k} has unexpected shape {raw_data_row_np.shape}. Expected ({_NUM_RAW_FEATURES},). Skipping correction.")
             self.Xpo[k] = self.Xpr[k].copy() # Ensure state propagation
             self.Ppo[k] = self.Ppr[k].copy()
             self.R = self.R_list[k] # Use prior R
             return

        self.input_buffer.append(raw_data_row_np)
        if len(self.input_buffer) > self.time_steps:
            self.input_buffer.pop(0)
        elif len(self.input_buffer) < self.time_steps:
            # Buffer not full, posterior remains prior
            self.Xpo[k] = self.Xpr[k].copy()
            self.Ppo[k] = self.Ppr[k].copy()
            self.R = self.R_list[k]
            return # Skip correction

        # --- Get LNN Prediction ---
        input_window_np = np.vstack(self.input_buffer)

        # Apply optimized preprocessing (with added checks)
        X_processed = self._preprocess_window(input_window_np)

        if X_processed is None:
            print(f"Warning: Preprocessing failed at step {k}. Skipping correction.")
            self.Xpo[k] = self.Xpr[k].copy()
            self.Ppo[k] = self.Ppr[k].copy()
            self.R = self.R_list[k]
            return

        # Check dimension mismatch again
        if X_processed.shape[1] != self.model_input_dim:
             print(f"Critical Error: Dimension mismatch at step {k}. Processed: {X_processed.shape[1]}, Model: {self.model_input_dim}. Skipping correction.")
             self.Xpo[k] = self.Xpr[k].copy()
             self.Ppo[k] = self.Ppr[k].copy()
             self.R = self.R_list[k]
             return

        X_tensor = torch.tensor(X_processed, dtype=torch.float32).unsqueeze(0).to(self.device)

        # LNN Inference
        try:
            with torch.no_grad():
                pred_scaled = self.liquid_model(X_tensor).cpu().numpy()
        except Exception as e:
            print(f"Error during LNN inference at step {k}: {e}. Skipping correction.")
            self.Xpo[k] = self.Xpr[k].copy(); self.Ppo[k] = self.Ppr[k].copy(); self.R = self.R_list[k]
            return

        # Inverse scale
        try:
            pred_pose = self.scaler_y.inverse_transform(pred_scaled)[0]
        except Exception as e:
            print(f"Error during inverse scaling at step {k}: {e}. Skipping correction.")
            self.Xpo[k] = self.Xpr[k].copy(); self.Ppo[k] = self.Ppr[k].copy(); self.R = self.R_list[k]
            return

        # Extract prediction
        pred_pos = pred_pose[:3]
        pred_quat_xyzw = pred_pose[3:]
        # Ensure robust quaternion creation even if values are slightly off
        try:
             q_pred = Quaternion(pred_quat_xyzw[3], pred_quat_xyzw[0], pred_quat_xyzw[1], pred_quat_xyzw[2])
             q_pred = q_pred.normalised
        except ValueError as e:
             print(f"Error creating prediction quaternion at step {k}: {e}. Values: {pred_quat_xyzw}. Skipping correction.")
             self.Xpo[k] = self.Xpr[k].copy(); self.Ppo[k] = self.Ppr[k].copy(); self.R = self.R_list[k]
             return

        # --- Calculate Residuals ---
        p_k_pr = self.Xpr[k, :3]
        v_k_pr = self.Xpr[k, 3:6] # Used in state update
        q_k_pr = Quaternion(self.q_list[k, :])

        pos_residual = pred_pos - p_k_pr

        q_diff = q_k_pr.conjugate * q_pred
        angle = q_diff.angle
        if abs(angle) < 1e-9:
            ang_residual = np.zeros(3)
        # (Keep handling for 180 deg as before, maybe add more robust axis finding if needed)
        elif abs(angle - math.pi) < 1e-9:
            print(f"Warning: Near 180 deg rotation difference at step {k}.")
            axis = q_diff.axis # axis method should handle normalization
            if np.linalg.norm(axis) < 1e-9: # Check if axis is valid
                 print(f"Error: Zero axis for near-pi rotation at step {k}")
                 ang_residual = np.zeros(3)
            else:
                 ang_residual = angle * axis
        else:
             axis = q_diff.axis
             ang_residual = angle * axis

        residual = np.concatenate([pos_residual, ang_residual])

        # --- Measurement Gating ---
        d_m = np.linalg.norm(pos_residual)
        GATE_THRESHOLD = 5.0

        if d_m >= GATE_THRESHOLD:
            print(f"Info: Measurement gated at step {k}, d_m = {d_m:.2f} >= {GATE_THRESHOLD}")
            self.Xpo[k] = self.Xpr[k].copy(); self.Ppo[k] = self.Ppr[k].copy(); self.R = self.R_list[k]
            return

        # --- Kalman Update ---
        H = np.zeros((6, 9)); H[:3, :3] = np.eye(3); H[3:6, 6:9] = np.eye(3)
        R_meas = np.diag([0.1**2, 0.1**2, 0.1**2, 0.05**2, 0.05**2, 0.05**2])
        P_k_pr = self.Ppr[k]
        S = H @ P_k_pr @ H.T + R_meas

        try:
            K = P_k_pr @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print(f"Error: Matrix S is singular at step {k}. Skipping update.")
            self.Xpo[k] = self.Xpr[k].copy(); self.Ppo[k] = self.Ppr[k].copy(); self.R = self.R_list[k]
            return

        delta_x = K @ residual

        # --- Inject correction ---
        self.Xpo[k, :3] = p_k_pr + delta_x[:3]
        self.Xpo[k, 3:6] = v_k_pr + delta_x[3:6] # Apply velocity correction

        delta_theta = delta_x[6:9]
        delta_q = Quaternion(self.zeta(delta_theta))
        q_k_po = q_k_pr * delta_q
        q_k_po = q_k_po.normalised
        self.q_list[k, :] = q_k_po.elements

        # --- Update covariance ---
        I_KH = np.eye(9) - K @ H
        P_k_po = I_KH @ P_k_pr @ I_KH.T + K @ R_meas @ K.T
        self.Ppo[k] = 0.5 * (P_k_po + P_k_po.T)

        # Update self.R for next prediction
        self.R = q_k_po.rotation_matrix
        self.R_list[k] = self.R

    # --- Helper methods (cross, zeta) remain unchanged ---
    def cross(self, v):
        """ Calculates the skew-symmetric matrix of a 3D vector v """
        v = np.squeeze(v)
        if v.shape != (3,):
            # Adding a print here might help debug if non-3D vectors are passed
            print(f"Error in cross: Input vector v has unexpected shape {v.shape}")
            raise ValueError(f"Input vector v must have 3 elements, but got shape {v.shape}")
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def zeta(self, phi):
        """ Converts an angle increment vector phi (rotation vector) into a unit quaternion delta_q """
        phi = np.squeeze(phi)
        if phi.shape != (3,):
             # Adding a print here might help debug if non-3D vectors are passed
            print(f"Error in zeta: Input vector phi has unexpected shape {phi.shape}")
            raise ValueError(f"Input vector phi must have 3 elements, but got shape {phi.shape}")
        phi_norm = np.linalg.norm(phi)
        if phi_norm < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = phi / phi_norm
        half_angle = 0.5 * phi_norm
        w = math.cos(half_angle)
        xyz = axis * math.sin(half_angle)
        return np.array([w, xyz[0], xyz[1], xyz[2]])