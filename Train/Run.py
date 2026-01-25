## configuration
import matplotlib.pyplot as plt
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ========== 随机种子 ==========
SEED = 1234
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 参数设置 ==========
epochs = 5000
batch_size = 128
learning_rate = 0.00005
test_size = 0.1
d_model = 512
n_head = 1
alpha = 0.9

from DataPre import Data_Pre
import os
from scipy.io import savemat

# === 导入双塔模型与训练器 ===
from img_tower import DSNetSingleStage         # 图像塔
from neural_tower import AttentionWithNoDeconvPos    # 神经塔
from ModelTrainer import DualTowerTrainer            # 训练器

# === 导入双塔 Tester ===
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# ---- 自定义 SSIM 函数 ----
def ssim_map(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2)
    return (ssim_n / ssim_d).clamp(0, 1)

def ssim_score(x, y):
    return ssim_map(x, y).mean().item()


def main():
    current_file_path = __file__
    top_directory = os.path.dirname(current_file_path)
    sub_dirs = ['M6']

    model_dir = 'Model'
    model_path = os.path.join(top_directory, model_dir)
    os.makedirs(model_path, exist_ok=True)

    for sub_dir in sub_dirs:
        data_path = os.path.join(top_directory, sub_dir, 'Data')

        # 读取数据
        data_x, data_y = Data_Pre(data_path)

        from ModelTrainer import setup_seed
        g, worker_fn = setup_seed(SEED)
        print(f'{sub_dir}_{data_x.shape[1]} neurons')
        print("原始数据维度：Rsp", data_x.shape, " Images", data_y.shape)



        # === 训练 ===
        trainer = DualTowerTrainer(
            img_data=data_y,
            neu_data=data_x,
            img_model_class=DSNetSingleStage,
            neu_model_class=AttentionWithNoDeconvPos,
            device=device.type,
            epochs=epochs,
            test_size=test_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            img_embed_dim=64,
            neu_d_model=d_model,
            neu_n_head=n_head,
            seed=SEED,
            alpha=alpha,
            external_generator=g,  # ✅ 传入随机源
            external_worker_init=worker_fn,  # ✅ 传入初始化函数
            sub_dir=sub_dir
        )


        resume_training = False
        if resume_training:
            img_ckpt_path = os.path.join(model_path, f"{sub_dir}_img_tower.pth")
            neu_ckpt_path = os.path.join(model_path, f"{sub_dir}_neu_tower.pth")
            if os.path.exists(img_ckpt_path) and os.path.exists(neu_ckpt_path):
                print(f"🔁 从 {img_ckpt_path} 和 {neu_ckpt_path} 加载已有模型进行继续训练...")
                trainer.img_checkpoint = img_ckpt_path
                trainer.neu_checkpoint = neu_ckpt_path
            else:
                print("⚠️ 未找到已有模型，重新训练。")
        else:
            # 🔧 清理旧的 checkpoint 引用（防止被意外加载）
            if hasattr(trainer, "img_checkpoint"):
                del trainer.img_checkpoint
            if hasattr(trainer, "neu_checkpoint"):
                del trainer.neu_checkpoint
            print("🚀 启动全新训练（未加载旧模型权重）")

        (best_img_model, best_neu_model), train_loss, val_loss = trainer.train()

        # === 保存权重 ===
        img_model_save_path = os.path.join(model_path, f'{sub_dir}_img_tower.pth')
        neu_model_save_path = os.path.join(model_path, f'{sub_dir}_neu_tower.pth')
        torch.save(best_img_model.state_dict(), img_model_save_path)
        torch.save(best_neu_model.state_dict(), neu_model_save_path)
        print(f"✅ Saved: {img_model_save_path}")
        print(f"✅ Saved: {neu_model_save_path}")

        # === 保存训练/验证损失 ===
        loss_save_path = os.path.join(top_directory, sub_dir)
        os.makedirs(loss_save_path, exist_ok=True)
        savemat(os.path.join(loss_save_path, 'loss_data.mat'),
                {'train_loss': np.array(train_loss), 'val_loss': np.array(val_loss)})

        print(f"子目录 {sub_dir} 训练完成。")

        # =====================================================
        # ✅ 双塔 Tester：计算前100个样本的特征图与 SSIM
        # =====================================================
        print("开始双塔测试 (取前100个样本)...")
        best_img_model.eval()
        best_neu_model.eval()

        imgs_100 = torch.tensor(data_y[:100], dtype=torch.float32).to(device)
        rsps_100 = torch.tensor(data_x[:100], dtype=torch.float32).to(device)

        z_img_all, z_neu_all = [], []
        ssim_list = []

        with torch.no_grad():
            for i in range(0, len(imgs_100), 8):
                img_batch = imgs_100[i:i + 8]
                rsp_batch = rsps_100[i:i + 8]

                # ---- 安全地取出第一个返回值 ----
                z_img_out = best_img_model(img_batch)
                z_img = z_img_out[0] if isinstance(z_img_out, tuple) else z_img_out

                z_neu_out = best_neu_model(rsp_batch)
                z_neu = z_neu_out[0] if isinstance(z_neu_out, tuple) else z_neu_out

                # ---- squeeze 处理 ----
                if z_img.ndim == 4 and z_img.shape[1] == 1:
                    z_img = z_img.squeeze(1)
                if z_neu.ndim == 4 and z_neu.shape[1] == 1:
                    z_neu = z_neu.squeeze(1)

                z_img_all.append(z_img.cpu())
                z_neu_all.append(z_neu.cpu())

                for b in range(z_img.size(0)):
                    ssim_val = ssim_score(z_img[b:b + 1].unsqueeze(1), z_neu[b:b + 1].unsqueeze(1))
                    ssim_list.append(ssim_val)

        z_img_all = torch.cat(z_img_all, dim=0)
        z_neu_all = torch.cat(z_neu_all, dim=0)
        mean_ssim = np.mean(ssim_list)

        print(f"✅ 平均结构相似度 SSIM: {mean_ssim:.4f}")
        print(f"前10个样本 SSIM: {ssim_list[:10]}")

        # === 保存测试结果 ===
        test_save_path = os.path.join(loss_save_path, 'test_result.mat')
        savemat(test_save_path, {
            'z_img': z_img_all.numpy(),
            'z_neu': z_neu_all.numpy(),
            'ssim_list': np.array(ssim_list),
            'mean_ssim': mean_ssim,
        })
        print(f"✅ 测试结果已保存到 {test_save_path}")


if __name__ == '__main__':
    main()
