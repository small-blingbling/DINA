import os
import torch
import numpy as np
import hdf5storage
from sklearn.preprocessing import MinMaxScaler

def Data_Pre(data_path):

    # ===== 1. 加载自然场景 =====
    img_file = os.path.join(data_path, 'images.mat')
    rsp_file = os.path.join(data_path, 'responses.mat')

    data_FigAll = hdf5storage.loadmat(img_file)
    target = data_FigAll['images']              # [H, W, N] 或 [N, H, W] 视数据而定
    target = target[:, :, :]                         # 保留所有 trial
    target = np.expand_dims(target, axis=1)          # [N, 1, H, W]

    # ===== 2. 加载神经响应 =====
    data_Response = hdf5storage.loadmat(rsp_file)
    source = data_Response['responses']                    # [num_neurons, N] or [N, num_neurons]
    source = source[:, :]
    # source = source.T                                # [N, num_neurons]

    # ===== 3. 归一化 =====
    scaler = MinMaxScaler()
    s = scaler.fit_transform(source)

    # ===== 4. 转为 tensor =====
    s = torch.tensor(s, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    # if target.shape[1] == 1:
    #     target = target.repeat(1, 3, 1, 1)           # [N, 3, H, W]

    print(f"✅ 数据加载成功: Rsp {s.shape}, Image {target.shape}")
    return s, target
