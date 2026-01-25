import torch
import numpy as np
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import torch.nn.functional as F
import random, copy, os
import torch.nn as nn

# ============================================================
# ✅ 随机种子固定函数
# ============================================================
def setup_seed(seed=1234):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return g, seed_worker


# ============================================================
# ✅ SSIM + MSE 混合损失（确定性版）
# ============================================================
def ssim_loss(x, y, alpha=1, C1=0.01 ** 2, C2=0.03 ** 2, eps=1e-8):
    if x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    if y.ndim == 4 and y.shape[1] == 1:
        y = y.squeeze(1)

    def normalize_feature(t):
        min_val = t.amin(dim=(1, 2), keepdim=True)
        max_val = t.amax(dim=(1, 2), keepdim=True)
        return (t - min_val) / (max_val - min_val + eps)

    x_n, y_n = normalize_feature(x), normalize_feature(y)
    x_n, y_n = x_n.unsqueeze(1), y_n.unsqueeze(1)

    mu_x = F.avg_pool2d(x_n, 3, 1, 1)
    mu_y = F.avg_pool2d(y_n, 3, 1, 1)
    sigma_x = F.avg_pool2d(x_n * x_n, 3, 1, 1) - mu_x.pow(2)
    sigma_y = F.avg_pool2d(y_n * y_n, 3, 1, 1) - mu_y.pow(2)
    sigma_xy = F.avg_pool2d(x_n * y_n, 3, 1, 1) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2)
    )
    ssim_val = ssim_map.mean()
    ssim_l = 1 - ssim_val
    mse_l = F.mse_loss(x, y)
    total_loss = alpha * ssim_l + (1 - alpha) * mse_l
    return total_loss

def normalize_per_sample(t, eps=1e-8):
    if t.ndim == 4 and t.shape[1] == 1:
        t = t.squeeze(1)
    B = t.shape[0]
    min_vals = t.view(B, -1).min(dim=1)[0].view(B, 1, 1)
    max_vals = t.view(B, -1).max(dim=1)[0].view(B, 1, 1)
    t_norm = (t - min_vals) / (max_vals - min_vals + eps)
    return t_norm.unsqueeze(1) * 255


# class CLIPLoss(nn.Module):
#     def __init__(self, init_temp=0.07):
#         super().__init__()
#         self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / init_temp)))
#
#     def forward(self, z_img, z_neu):
#         z_img = F.normalize(z_img.flatten(1), dim=-1)
#         z_neu = F.normalize(z_neu.flatten(1), dim=-1)
#         logits = z_img @ z_neu.t() * self.logit_scale.exp()
#         labels = torch.arange(z_img.size(0), device=z_img.device)
#         loss_i = F.cross_entropy(logits, labels)
#         loss_t = F.cross_entropy(logits.t(), labels)
#         return (loss_i + loss_t) / 2


import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as sio

class CLIPLoss(nn.Module):
    def __init__(self, init_temp=0.01, w_clip=1.0, w_rmse=0.5, w_ssim=0.5):
        """
        混合损失函数:
        total_loss = w_clip * CLIP_loss + w_rmse * RMSE_loss + w_ssim * SSIM_loss
        """
        super().__init__()
        self.logit_scale = torch.log(torch.tensor(1.0 / init_temp))

        self.w_clip = w_clip
        self.w_rmse = w_rmse
        self.w_ssim = w_ssim

    def forward(self, z_img, z_neu):
        # ========= 1️⃣ CLIP 对比损失 =========
        z_img_flat = F.normalize(z_img.flatten(1), dim=-1)
        z_neu_flat = F.normalize(z_neu.flatten(1), dim=-1)
        logits = z_img_flat @ z_neu_flat.t() * self.logit_scale.exp()
        labels = torch.arange(z_img_flat.size(0), device=z_img.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        clip_loss = (loss_i + loss_t) / 2

        # ========= 2️⃣ RMSE 损失 =========
        rmse_loss = torch.sqrt(F.mse_loss(z_img, z_neu))

        # ========= 3️⃣ SSIM 损失 =========
        ssim_loss_val = 1 - self.ssim(z_img, z_neu)

        # ========= 4️⃣ 加权融合 =========
        total_loss = (
            self.w_clip * clip_loss +
            self.w_rmse * rmse_loss +
            self.w_ssim * ssim_loss_val
        )


        # print(rmse_loss)
        return total_loss

    # ========= 辅助函数: SSIM =========
    def ssim(self, x, y, C1=0.01 ** 2, C2=0.03 ** 2, eps=1e-8):
        """
        简化版 SSIM（单通道或多通道均可）
        """
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
            y = y.squeeze(1)

        def normalize(t):
            min_t = t.amin(dim=(1, 2), keepdim=True)
            max_t = t.amax(dim=(1, 2), keepdim=True)
            return (t - min_t) / (max_t - min_t + eps)

        x = normalize(x)
        y = normalize(y)
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(y * y, 3, 1, 1) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2)
        )
        return ssim_map.mean()



# ============================================================
# ✅ 双塔训练器（增加冻结与手动解冻功能）
# ============================================================
class DualTowerTrainer:
    def __init__(self, img_data, neu_data,
                 img_model_class, neu_model_class,
                 device='cpu',
                 epochs=2000, test_size=0.1, batch_size=8,
                 learning_rate=0.00005,
                 img_embed_dim=64, neu_d_model=128, neu_n_head=4,
                 seed=1234, alpha=0.9,
                 external_generator=None, external_worker_init=None,sub_dir=None):
        self.img_data = img_data
        self.neu_data = neu_data
        self.img_model_class = img_model_class
        self.neu_model_class = neu_model_class
        self.device = device
        self.epochs = epochs
        self.test_size = test_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.img_embed_dim = img_embed_dim
        self.neu_d_model = neu_d_model
        self.neu_n_head = neu_n_head
        self.seed = seed
        self.alpha = alpha
        self.sub_dir = sub_dir

        if external_generator is not None:
            self.generator = external_generator
            self.seed_worker = external_worker_init
        else:
            self.generator, self.seed_worker = setup_seed(seed)

        self.train_loader, self.test_loader = self.prepare_data_loaders()

    def prepare_data_loaders(self):
        dataset = DualDataSet(self.img_data, self.neu_data)
        train_data, test_data = train_test_split(dataset, test_size=self.test_size, random_state=self.seed)
        train_loader = Data.DataLoader(train_data, batch_size=self.batch_size,
                                       shuffle=True, drop_last=False,
                                       worker_init_fn=self.seed_worker,
                                       generator=self.generator)
        test_loader = Data.DataLoader(test_data, batch_size=self.batch_size,
                                      shuffle=False, drop_last=False,
                                      worker_init_fn=self.seed_worker,
                                      generator=self.generator)
        print("TestDataLoader 的 batch 个数", test_loader.__len__())
        print("TrainDataLoader 的 batch 个数", train_loader.__len__())
        return train_loader, test_loader

    def eval_test(self, img_model, neu_model):
        test_losses = []
        with torch.no_grad():
            for imgs, rsps in self.test_loader:
                imgs, rsps = imgs.to(self.device), rsps.to(self.device)
                z_img = img_model(imgs)
                z_neu, *_ = neu_model(rsps)
                criterion = CLIPLoss(w_clip=1, w_rmse=0, w_ssim=0)
                loss = criterion(z_img, z_neu)

                test_losses.append(loss.item())
        return np.mean(test_losses)

    # def train(self):
    #     # img_model = self.img_model_class(
    #     #     img_size=100, in_chans=1,
    #     #     embed_dim=self.img_embed_dim,
    #     #     depth=2, num_heads=4,
    #     #     mlp_ratio=4.0,
    #     #     drop_rate=0., attn_drop_rate=0., drop_path_rate=0.
    #     # ).to(self.device)
    #
    #     img_model = self.img_model_class().to(self.device)
    #
    #     neu_model = self.neu_model_class(
    #         input_size=self.neu_data.shape[1],
    #         d_model=self.neu_d_model,
    #         n_head=self.neu_n_head,
    #         max_len=self.neu_data.shape[1],
    #         num_neurons=self.neu_data.shape[1],
    #         device=self.device,
    #         hidden=1024
    #     ).to(self.device)
    #
    #     # ==== 如果检测到热启动标志，加载已有权重 ====
    #     if hasattr(self, 'img_checkpoint') and os.path.exists(self.img_checkpoint):
    #         img_model.load_state_dict(torch.load(self.img_checkpoint, map_location=self.device))
    #         print("✅ 图像塔权重已加载。")
    #
    #     if hasattr(self, 'neu_checkpoint') and os.path.exists(self.neu_checkpoint):
    #         neu_model.load_state_dict(torch.load(self.neu_checkpoint, map_location=self.device))
    #         print("✅ 神经塔权重已加载。")
    #
    #     # ==== ✅ 冻结图像塔的预训练部分 ====
    #     if hasattr(img_model, "base"):
    #         for p in img_model.base.parameters():
    #             p.requires_grad = True
    #         print("🧊 冻结图像塔预训练参数，只训练拼接层。")
    #
    #     # === 定义解冻函数 ===
    #     def unfreeze_stages(model, stages_to_unfreeze):
    #         for name, param in model.base.named_parameters():
    #             stage = None
    #             for i in range(1, 5):
    #                 if f"blocks{i}" in name or f"patch_embed{i}" in name:
    #                     stage = i
    #                     break
    #             if stage in stages_to_unfreeze:
    #                 param.requires_grad = True
    #
    #     # ==== 优化器定义（只训练可更新参数） ====
    #     trainable_params = [p for p in img_model.parameters() if p.requires_grad]
    #     opt_img = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=5e-5)
    #
    #     opt_neu = torch.optim.RMSprop(
    #         neu_model.parameters(),
    #         lr=2e-4,
    #         alpha=0.95
    #     )
    #
    #     sched_img = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_img, mode='min', factor=0.5, patience=50)
    #     sched_neu = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_neu, mode='min', factor=0.5, patience=50)
    #
    #     best_test_loss = float('inf')
    #     train_loss_hist, test_loss_hist = [], []
    #
    #     print("开始双塔训练...")
    #     try:
    #         for epoch in range(self.epochs):
    #
    #             # if epoch == 100000:
    #             #     unfreeze_stages(img_model, [3])
    #             #     print("🟢 已解冻 stage3")
    #
    #
    #             # 每次解冻后，重新构建优化器，保证新参数加入训练
    #             trainable_params = [p for p in img_model.parameters() if p.requires_grad]
    #             opt_img = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=5e-5)
    #
    #             img_model.train()
    #             neu_model.train()
    #             epoch_losses = []
    #
    #             for imgs, rsps in self.train_loader:
    #                 imgs, rsps = imgs.to(self.device), rsps.to(self.device)
    #                 z_img = img_model(imgs)
    #                 z_neu, *_ = neu_model(rsps)
    #                 criterion = CLIPLoss(w_clip=1, w_rmse=0, w_ssim=0)
    #                 loss = criterion(z_img, z_neu)
    #
    #                 opt_img.zero_grad()
    #                 opt_neu.zero_grad()
    #                 loss.backward()
    #                 opt_img.step()
    #                 opt_neu.step()
    #                 epoch_losses.append(loss.item())
    #
    #             train_loss = np.mean(epoch_losses)
    #             test_loss = self.eval_test(img_model, neu_model)
    #             train_loss_hist.append(train_loss)
    #             test_loss_hist.append(test_loss)
    #
    #             sched_img.step(test_loss)
    #             sched_neu.step(test_loss)
    #
    #             if test_loss < best_test_loss:
    #                 best_test_loss = test_loss
    #                 best_models = (
    #                     copy.deepcopy(img_model.state_dict()),
    #                     copy.deepcopy(neu_model.state_dict())
    #                 )
    #                 print("best_model...")
    #
    #             img_model.load_state_dict(best_models[0])
    #             neu_model.load_state_dict(best_models[1])
    #
    #             print(f"Epoch {epoch + 1}/{self.epochs} | Train: {train_loss:.6f} | Test: {test_loss:.6f}")
    #
    #     except KeyboardInterrupt:
    #         print("\n手动中断训练，保存当前最佳模型...")
    #
    #     return (img_model, neu_model), train_loss_hist, test_loss_hist

    def train(self):
        print("⚙️ 初始化模型...")

        # -----------------------
        # 1. 初始化模型
        # -----------------------
        img_model = self.img_model_class().to(self.device)
        neu_model = self.neu_model_class(
            input_size=self.neu_data.shape[1],
            d_model=self.neu_d_model,
            n_head=self.neu_n_head,
            max_len=self.neu_data.shape[1],
            num_neurons=self.neu_data.shape[1],
            device=self.device,
            hidden=1024
        ).to(self.device)

        # -----------------------
        # 2. 加载 checkpoint (如需要)
        # -----------------------
        if hasattr(self, 'img_checkpoint') and os.path.exists(self.img_checkpoint):
            img_model.load_state_dict(torch.load(self.img_checkpoint, map_location=self.device))
            print("✅ 图像塔权重已加载")

        if hasattr(self, 'neu_checkpoint') and os.path.exists(self.neu_checkpoint):
            neu_model.load_state_dict(torch.load(self.neu_checkpoint, map_location=self.device))
            print("✅ 神经塔权重已加载")

        # -----------------------
        # 3. 冻结 or 解冻 base
        # -----------------------
        if hasattr(img_model, "base"):
            for p in img_model.base.parameters():
                p.requires_grad = True  # 如需冻结改 False
            print("🧊 图像塔 base 已设为可训练（如需冻结改为 False）")

        # -----------------------
        # 4. 创建优化器（⚠️必须放在 epoch 外）
        # -----------------------
        img_params = [p for p in img_model.parameters() if p.requires_grad]
        opt_img = torch.optim.AdamW(img_params, lr=3e-4, weight_decay=5e-5)

        opt_neu = torch.optim.RMSprop(
            neu_model.parameters(),
            lr=5e-5,
            alpha=0.9
        )

        # 调度器
        sched_img = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_img, mode='min', factor=0.5, patience=50)
        sched_neu = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_neu, mode='min', factor=0.5, patience=50)

        criterion = CLIPLoss(w_clip=1, w_rmse=0, w_ssim=0)

        best_test_loss = float('inf')
        best_models = None
        train_loss_hist, test_loss_hist = [], []

        print("开始训练...")

        try:
            for epoch in range(self.epochs):

                img_model.train()
                neu_model.train()

                epoch_losses = []

                for imgs, rsps in self.train_loader:
                    imgs, rsps = imgs.to(self.device), rsps.to(self.device)

                    z_img = img_model(imgs)
                    z_neu, *_ = neu_model(rsps)

                    loss = criterion(z_img, z_neu)

                    opt_img.zero_grad(set_to_none=True)
                    opt_neu.zero_grad(set_to_none=True)

                    loss.backward()

                    opt_img.step()
                    opt_neu.step()

                    epoch_losses.append(loss.item())

                train_loss = np.mean(epoch_losses)
                test_loss = self.eval_test(img_model, neu_model)

                train_loss_hist.append(train_loss)
                test_loss_hist.append(test_loss)

                sched_img.step(test_loss)
                sched_neu.step(test_loss)

                # if test_loss < best_test_loss:
                #     best_test_loss = test_loss
                #     best_models = (
                #         copy.deepcopy(img_model.state_dict()),
                #         copy.deepcopy(neu_model.state_dict())
                #     )
                #     print("best_model...")

                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_models = (
                        copy.deepcopy(img_model.state_dict()),
                        copy.deepcopy(neu_model.state_dict())
                    )
                    print("best_model...")

                    # ====== 立即保存到 Model 文件夹 ======
                    save_dir = "./Model"
                    os.makedirs(save_dir, exist_ok=True)

                    # img_save_path = os.path.join(save_dir, "best_img_tower.pth")
                    # neu_save_path = os.path.join(save_dir, "best_neu_tower.pth")

                    img_save_path = os.path.join(save_dir, f"{self.sub_dir}_best_img_tower.pth")
                    neu_save_path = os.path.join(save_dir, f"{self.sub_dir}_best_neu_tower.pth")

                    torch.save(best_models[0], img_save_path)
                    torch.save(best_models[1], neu_save_path)

                    loss_dir = os.path.join(".", self.sub_dir)
                    os.makedirs(loss_dir, exist_ok=True)

                    loss_save_path = os.path.join(loss_dir, "loss_data.mat")
                    sio.savemat(loss_save_path, {
                        "train_loss_hist": np.array(train_loss_hist),
                        "test_loss_hist": np.array(test_loss_hist)
                    })

                print(f"Epoch {epoch + 1}/{self.epochs} | Train {train_loss:.4f} | Test {test_loss:.4f}")

        except KeyboardInterrupt:
            print("\n检测到手动中断，正在恢复最佳模型...")



        return (img_model, neu_model), train_loss_hist, test_loss_hist


# ============================================================
# ✅ 数据集封装
# ============================================================
class DualDataSet(Data.Dataset):
    def __init__(self, img_data, neu_data):
        assert len(img_data) == len(neu_data)
        self.imgs = torch.FloatTensor(img_data)
        self.rsps = torch.FloatTensor(neu_data)

    def __getitem__(self, index):
        return self.imgs[index], self.rsps[index]

    def __len__(self):
        return len(self.imgs)
