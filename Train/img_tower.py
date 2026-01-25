import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ds_net_small   # 🔥 调用你修复后的 DS-Net small


class DSNetSingleStage(nn.Module):
    def __init__(self, stage=2, block=4):
        super().__init__()
        self.stage = stage
        self.block = block

        self.base = ds_net_small(img_size=68, num_classes=0, in_chans=1)

        # 注册 hook
        self._register_target_hook()

        # 保存 hook 特征
        self.features = {}

        self.my_conv = nn.Conv2d(128, 1, kernel_size=1)


    def _register_target_hook(self):
        target_name = f"blocks{self.stage}.{self.block - 1}"
        print(f"✅ Hook 注册到: {target_name}")

        module_dict = dict(self.base.named_modules())
        assert target_name in module_dict, f"❌ 未找到模块 {target_name}"

        def hook_fn(module, input, output):
            self.features["feat"] = output.detach()

        module_dict[target_name].register_forward_hook(hook_fn)


    def forward(self, x):

        self.features.clear()

        _ = self.base(x)

        assert "feat" in self.features, "❌ Hook 未触发（检查 stage 和 block 是否匹配）"

        feat = self.features["feat"]

        # 320-ch → 1-ch
        feat = self.my_conv(feat)

        # 上采样成 32×32
        feat = F.interpolate(feat, size=(16, 64), mode="bilinear")

        # 归一化
        feat_min = feat.amin(dim=(2, 3), keepdim=True)
        feat_max = feat.amax(dim=(2, 3), keepdim=True)
        feat = (feat - feat_min) / (feat_max - feat_min + 1e-8)

        return feat



# =============================
# 🔍 测试
# =============================
if __name__ == "__main__":
    model = DSNetSingleStage(stage=3, block=9)  # 🔥 指定使用 blocks3[7]
    model.eval()

    x = torch.randn(10, 1, 68, 270)  # 输入单通道自然图
    out = model(x)
    print("输出结果:", out.shape)
