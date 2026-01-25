# import torch
# import torch.nn as nn
# import math
# import torch.nn.functional as F
#
# class Embedding(nn.Module):
#     def __init__(self, d_model, num_neurons):
#         super(Embedding, self).__init__()
#         # self.embedding = nn.Linear(1,d_model)
#         self.embedding_layers = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_neurons)])
#
#     def forward(self, x):
#         # x = self.embedding(x)
#         x = torch.stack([layer(x[:, i, :]) for i, layer in enumerate(self.embedding_layers)], dim=1)
#         return x
#
# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model, max_len, device):
#
#         super(PositionalEncoding, self).__init__()
#
#         self.encoding = torch.zeros(max_len, d_model, device=device)
#         self.encoding.requires_grad = False  # we don't need to compute gradient
#
#         pos = torch.arange(0, max_len, device=device)
#         pos = pos.float().unsqueeze(dim=1)
#
#         _2i = torch.arange(0, d_model, step=2, device=device).float()
#
#         self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
#         self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
#
#     def forward(self, x):
#         batch_size, seq_len = x.size()
#         return self.encoding[:seq_len, :]
#
#
# class ScaleDotProductAttention(nn.Module):
#
#     def __init__(self):
#         super(ScaleDotProductAttention, self).__init__()
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, q, k, v, mask=None, e=1e-12):
#         batch_size, head, length, d_tensor = k.size()
#
#         k_t = k.transpose(2, 3)  # transpose
#         score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
#         if mask is not None:
#             score = score.masked_fill(mask == 0, -10000)
#         score = self.softmax(score)
#         v = score @ v
#
#         return v, score
#
# class MultiHeadAttention(nn.Module):
#
#     def __init__(self, d_model, n_head):
#         super(MultiHeadAttention, self).__init__()
#         self.n_head = n_head
#         self.attention = ScaleDotProductAttention()
#         self.w_q = nn.Linear(d_model, d_model)
#         self.w_k = nn.Linear(d_model, d_model)
#         self.w_v = nn.Linear(d_model, d_model)
#         self.w_concat = nn.Linear(d_model, d_model)
#
#     def forward(self, q, k, v, mask=None):
#         # 1. dot product with weight matrices
#         q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
#
#         # 2. split tensor by number of heads
#         q, k, v = self.split(q), self.split(k), self.split(v)
#
#         # 3. do scale dot product to compute similarity
#         out, attention = self.attention(q, k, v, mask=mask)
#
#         # 4. concat and pass to linear layer
#         out = self.concat(out)
#         out = self.w_concat(out)
#
#
#         return out,attention
#
#     def split(self, tensor):
#
#         batch_size, length, d_model = tensor.size()
#
#         d_tensor = d_model // self.n_head
#         tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
#
#         return tensor
#
#     def concat(self, tensor):
#
#         batch_size, head, length, d_tensor = tensor.size()
#         d_model = head * d_tensor
#
#         tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
#         return tensor
#
# class unEmbedding(nn.Module):
#     def __init__(self, d_model,num_neurons):
#
#         super(unEmbedding, self).__init__()
#         # self.embedding = nn.Linear(d_model,1)
#         self.embedding_layers = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_neurons)])
#
#
#     def forward(self, x):
#         # x = self.embedding(x)
#         x = torch.stack([layer(x[:, i, :]) for i, layer in enumerate(self.embedding_layers)], dim=1)
#         return x
#
#
#
# class AttentionWithNoDeconvPos(nn.Module):
#     def __init__(self, input_size, d_model , n_head, max_len, num_neurons, device, hidden):
#         super(AttentionWithNoDeconvPos, self).__init__()
#
#         # self.EmbeddingWithPos = TransformerEmbedding(input_size, d_model, max_len, drop_prob, device)
#         self.embedding = Embedding(d_model, num_neurons)
#         self.pos = PositionalEncoding(d_model, max_len, device)
#         self.attention = MultiHeadAttention(d_model,n_head)
#         self.ff = unEmbedding(d_model, num_neurons)
#         # self.ff1 = nn.Linear(input_size, 64)
#         self.ff1 = nn.Sequential(
#             nn.Linear(input_size, 2048),
#             nn.Linear(2048, 512),
#             nn.Linear(512, 256)
#         )
#
#     def forward(self, x):
#
#         # x = self.EmbeddingWithPos(x)
#
#         # pos_x = self.posencoding(torch.zeros(2,1707))
#         x_pos = self.pos(x)
#         x = x.unsqueeze(2)
#         # print(x.shape)
#         x = self.embedding(x)
#
#         # x = x + x_pos
#
#         X = x
#         x , atten = self.attention(x,x,x)
#         x = self.ff(x)
#         x_atten = x
#         x = x.squeeze()
#         # print(x.shape)
#         x = self.ff1(x)
#         attention_weights = atten.squeeze()
#         # x = self.ff2(x)
#         # x = self.relu(x)
#         # height = x.size(1)
#         # h = math.sqrt(height)
#         # y = x.reshape(-1, 1, int(h), int(h))
#         # x = x.reshape(-1, 1, int(h), int(h))
#         x = x.reshape(-1, 1, 8, 32)
#         x = F.interpolate(x, size=(16, 64), mode="bilinear")
#
#         x_min = x.amin(dim=(2, 3), keepdim=True)
#         x_max = x.amax(dim=(2, 3), keepdim=True)
#         x = (x - x_min) / (x_max - x_min + 1e-8)
#
#         return x, attention_weights, x_atten


import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Embedding(nn.Module):
    def __init__(self, d_model, num_neurons):
        super(Embedding, self).__init__()
        # 保持原来结构（未改动）
        self.embedding_layers = nn.ModuleList([nn.Linear(1, d_model) for _ in range(num_neurons)])

    def forward(self, x):
        x = torch.stack([layer(x[:, i, :]) for i, layer in enumerate(self.embedding_layers)], dim=1)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


# ============================================================
# 🔥 修改重点：Flash Attention / SDPA + 可选返回注意力图
# ============================================================

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, need_weights=False):
        """
        自动兼容两种情况：
        1. 高版本 PyTorch (>= 2.0): 使用 Flash Attention
        2. 低版本 PyTorch: 使用普通 attention（手动 softmax）
        """

        use_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if use_sdpa and not need_weights:
            # ⚡ 高版本 PyTorch，训练阶段使用 Flash Attention
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
            return out, None

        elif use_sdpa and need_weights:

            out, attn = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, need_attn=True
            )
            return out, attn

        else:

            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

            if need_weights:
                return out, attn
            else:
                return out, None




class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, need_weights=False):

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        q, k, v = self.split(q), self.split(k), self.split(v)

        out, attn = self.attention(q, k, v, need_weights=need_weights)

        out = self.concat(out)
        out = self.w_concat(out)

        return out, attn


    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor

    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor



class unEmbedding(nn.Module):
    def __init__(self, d_model, num_neurons):
        super(unEmbedding, self).__init__()
        self.embedding_layers = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_neurons)])

    def forward(self, x):
        x = torch.stack([layer(x[:, i, :]) for i, layer in enumerate(self.embedding_layers)], dim=1)
        return x



class AttentionWithNoDeconvPos(nn.Module):
    def __init__(self, input_size, d_model , n_head, max_len, num_neurons, device, hidden):
        super(AttentionWithNoDeconvPos, self).__init__()

        self.embedding = Embedding(d_model, num_neurons)
        self.pos = PositionalEncoding(d_model, max_len, device)
        self.attention = MultiHeadAttention(d_model, n_head)
        self.ff = unEmbedding(d_model, num_neurons)

        self.ff1 = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Linear(2048, 512),
            nn.Linear(512, 256)
        )


    # ⭐ 增加参数 return_attn（默认 False）
    def forward(self, x, return_attn=False):

        x_pos = self.pos(x)

        x = x.unsqueeze(2)
        x = self.embedding(x)

        # 🎯 注意力：训练时不返回 attn，可视化时返回 attn
        x, atten = self.attention(x, x, x, need_weights=return_attn)

        x = self.ff(x)
        x_atten = x

        x = x.squeeze()

        x = self.ff1(x)

        attention_weights = None if atten is None else atten.squeeze()

        x = x.reshape(-1, 1, 8, 32)
        x = F.interpolate(x, size=(16, 64), mode="bilinear")

        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        x = (x - x_min) / (x_max - x_min + 1e-8)

        return x, attention_weights, x_atten
