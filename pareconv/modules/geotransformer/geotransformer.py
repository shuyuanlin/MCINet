# Reference: https://github.com/qinzheng93/GeoTransformer

import torch.nn as nn
from pareconv.modules.geotransformer.positional_encoding import GeometricStructureEmbedding
from pareconv.modules.geotransformer.ppftransformer import RPETransformerLayer, TransformerLayer
import torch
from torch.nn import init
import torch.nn.functional as F


def _check_block_type(block):
    if block not in ['self', 'cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))

class ContraNorm(nn.Module):
    def __init__(self, dim, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False):
        super().__init__()
        if learnable and scale > 0:
            import math
            if positive:
                scale_init = math.log(scale)
            else:
                scale_init = scale
            self.scale_param = nn.Parameter(torch.empty(dim).fill_(scale_init))
        self.dual_norm = dual_norm
        self.scale = scale
        self.pre_norm = pre_norm
        self.temp = temp
        self.learnable = learnable
        self.positive = positive
        self.identity = identity

        self.layernorm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        if self.scale > 0.0:
            xn = nn.functional.normalize(x, dim=2)
            if self.pre_norm:
                x = xn
            sim = torch.bmm(xn, xn.transpose(1,2)) / self.temp
            if self.dual_norm:
                sim = nn.functional.softmax(sim, dim=2) + nn.functional.softmax(sim, dim=1)
            else:
                sim = nn.functional.softmax(sim, dim=2)
            x_neg = torch.bmm(sim, x)
            if not self.learnable:
                if self.identity:
                    x = (1+self.scale) * x - self.scale * x_neg
                else:
                    x = x - self.scale * x_neg
            else:
                scale = torch.exp(self.scale_param) if self.positive else self.scale_param
                scale = scale.view(1, 1, -1)
                if self.identity:
                    x = scale * x - scale * x_neg
                else:
                    x = x - scale * x_neg
        x = self.layernorm(x)
        return x

# 定义外部注意力类，继承自nn.Module
class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        # 初始化两个线性变换层，用于生成注意力映射
        # mk: 将输入特征从d_model维映射到S维，即降维到共享内存空间的大小
        self.mk = nn.Linear(d_model, S, bias=False)
        # mv: 将降维后的特征从S维映射回原始的d_model维
        self.mv = nn.Linear(S, d_model, bias=False)
        # 使用Softmax函数进行归一化处理
        self.softmax = nn.Softmax(dim=1)
        # 调用权重初始化函数
        self.init_weights()

    def init_weights(self):
        # 自定义权重初始化方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对卷积层的权重进行Kaiming正态分布初始化
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    # 如果有偏置项，则将其初始化为0
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 对批归一化层的权重和偏置进行常数初始化
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对线性层的权重进行正态分布初始化，偏置项（如果存在）初始化为0
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        # 前向传播函数
        attn = self.mk(queries)  # 使用mk层将输入特征降维到S维
        attn = self.softmax(attn)  # 对降维后的特征进行Softmax归一化处理
        # 对归一化后的注意力分数进行标准化，使其和为1
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)  # 使用mv层将注意力特征映射回原始维度
        return out


class EfficientAdditiveAttnetion(nn.Module):
    """
    高效加性注意力模块，用于SwiftFormer中。
    输入：形状为[B, N, D]的张量
    输出：形状为[B, N, D]的张量
    """

    def __init__(self, in_dims=512, token_dim=512):
        super().__init__()
        # 初始化查询和键的线性变换
        self.to_query = nn.Linear(in_dims, token_dim)
        self.to_key = nn.Linear(in_dims, token_dim)

        # 初始化可学习的权重向量和缩放因子
        self.w_a = nn.Parameter(torch.randn(token_dim, 1))
        self.scale_factor = token_dim ** -0.5

        # 初始化后续的线性变换
        self.Proj = nn.Linear(token_dim, token_dim)
        self.final = nn.Linear(token_dim, token_dim)

    def forward(self, x):
        B, N, D = x.shape  # B:批次大小，N:序列长度，D:特征维度

        # 生成初步的查询和键矩阵
        query = self.to_query(x)
        key = self.to_key(x)

        # 对查询和键进行标准化处理
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        # 学习查询的注意力权重，并进行缩放和标准化
        query_weight = query @ self.w_a
        A = query_weight * self.scale_factor
        A = torch.nn.functional.normalize(A, dim=1)

        # 通过注意力权重对查询进行加权，以生成全局查询向量
        q = torch.sum(A * query, dim=1)
        q = q.reshape(B, 1, -1)

        # 计算全局查询向量和每个键的交互，再与原始查询进行逐元素相加
        out = self.Proj(q * key) + query
        out = self.final(out)  # 通过最终的线性层输出调制后的特征

        return out


class SimA(nn.Module):
    """ SimA attention block
    """

    def __init__(self, dim, num_heads=3, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        k = F.normalize(k, p=1.0, dim=-2)
        q = F.normalize(q, p=1.0, dim=-2)
        if N < (C // self.num_heads):
            x = ((q @ k.transpose(-2, -1)) @ v).transpose(1, 2).reshape(B, N, C)
        else:
            x = (q @ (k.transpose(-2, -1) @ v)).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RPEConditionalTransformer(nn.Module):
    def __init__(
            self,
            blocks,
            d_model,
            num_heads,
            dropout=None,
            activation_fn='ReLU',
            return_attention_scores=False,
            parallel=False,
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        # self.EAA = EfficientAdditiveAttnetion(192,192)
        # self.sima = SimA(192)
        # self.dat = DeformableAttention1D(dim=192, downsample_factor=4, offset_scale=2, offset_kernel_size=6)
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    # def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
    #     attention_scores = []
    #     for i, block in enumerate(self.blocks):
    #         if block == 'self':
    #             feats0, scores0, pos0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
    #             feats1, scores1, pos1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
    #         else:
    #             feats0, scores0 = self.layers[i](feats0, feats1, pos0, pos1, memory_masks=masks1)
    #             feats1, scores1 = self.layers[i](feats1, feats0, pos1, pos0, memory_masks=masks0)
    #
    #         if self.return_attention_scores:
    #             attention_scores.append([scores0, scores1])
    #     if self.return_attention_scores:
    #         return feats0, feats1, attention_scores
    #     else:
    #         return feats0, feats1

    # 改
    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        # print(feats0.shape)  # torch.Size([1, 113, 192])
        # print(feats1.shape)   # torch.Size([1, 252, 192])
        # add 高效注意力
        # feats0 = self.EAA(feats0)
        # feats1 = self.EAA(feats1)

        # 改
        # feats0 = self.sima(feats0)
        # feats1 = self.sima(feats1)

        #
        # feats0 = feats0.permute(0, 2, 1)
        # feats0 = self.dat(feats0)
        # feats0 = feats0.permute(0, 2, 1)
        #
        # feats1 = feats1.permute(0, 2, 1)
        # feats1 = self.dat(feats1)
        # feats1 = feats1.permute(0, 2, 1)

        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, embeddings0, embeddings1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, embeddings1, embeddings0, memory_masks=masks0)

            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class GeometricTransformer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dim,
            num_heads,
            blocks,
            sigma_d,
            sigma_a,
            angle_k,
            dropout=None,
            activation_fn='ReLU',
            reduction_a='max',
    ):
        r"""Geometric Transformer (GeoTransformer).
        Args:
            input_dim: input feature dimension
            output_dim: output feature dimension
            hidden_dim: hidden feature dimension
            num_heads: number of head in transformer
            blocks: list of 'self' or 'cross'
            sigma_d: temperature of distance
            sigma_a: temperature of angles
            angle_k: number of nearest neighbors for angular embedding
            activation_fn: activation function
            reduction_a: reduction mode of angular embedding ['max', 'mean']
        """
        super(GeometricTransformer, self).__init__()

        self.embedding = GeometricStructureEmbedding(hidden_dim, sigma_d, sigma_a, angle_k, reduction_a=reduction_a)

        self.in_proj = nn.Linear(input_dim, hidden_dim)
        # self.exter = ExternalAttention(768,8)
        self.transformer = RPEConditionalTransformer(
            blocks, hidden_dim, num_heads, dropout=dropout, activation_fn=activation_fn
        )
        # self.contranorm = ContraNorm(dim=192, scale=0.1, dual_norm=False, pre_norm=False, temp=1.0, learnable=False, positive=False, identity=False)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        # self.sima = SimA(768)

    def forward(
            self,
            ref_points,
            src_points,
            ref_feats,
            src_feats,
            ref_masks=None,
            src_masks=None,
    ):
        r"""Geometric Transformer
        Args:
            ref_points (Tensor): (B, N, 3)
            src_points (Tensor): (B, M, 3)
            ref_feats (Tensor): (B, N, C)
            src_feats (Tensor): (B, M, C)
            ref_masks (Optional[BoolTensor]): (B, N)
            src_masks (Optional[BoolTensor]): (B, M)
        Returns:
            ref_feats: torch.Tensor (B, N, C)
            src_feats: torch.Tensor (B, M, C)
        """
        ref_embeddings = self.embedding(ref_points)
        src_embeddings = self.embedding(src_points)

        # print(ref_feats.shape)  #torch.Size([1, 113, 768])
        # print(src_feats.shape)   #torch.Size([1, 252, 768])

        # add  外部注意力
        # ref_feats = self.exter(ref_feats)
        # src_feats = self.exter(src_feats)

        # add
        # ref_feats = self.sima(ref_feats)
        # ref_feats = self.sima(ref_feats)

        ref_feats = self.in_proj(ref_feats)
        # print(ref_feats.shape) #torch.Size([1, 113, 192])
        src_feats = self.in_proj(src_feats)

        ref_feats, src_feats = self.transformer(
            ref_feats,
            src_feats,
            ref_embeddings,
            src_embeddings,
            masks0=ref_masks,
            masks1=src_masks,
        )
        # ref_feats =self.contranorm(ref_feats)
        # src_feats =self.contranorm(src_feats)

        ref_feats = self.out_proj(ref_feats)
        src_feats = self.out_proj(src_feats)
        # ref_feats = self.sima(ref_feats)
        # src_feats = self.sima(src_feats)

        # print(ref_feats.shape)
        return ref_feats, src_feats


if __name__ == '__main__':
    ref_points = torch.rand(2, 5, 3, requires_grad=True)
    src_points = torch.rand(2, 5, 3, requires_grad=True)
    ref_feats = torch.rand(2, 5, 8, requires_grad=True)
    src_feats = torch.rand(2, 5, 8, requires_grad=True)

    # 初始化 GeometricTransformer
    geo_transformer = GeometricTransformer(
        input_dim=8,
        output_dim=8,
        hidden_dim=16,
        num_heads=4,
        blocks=['self', 'cross'],
        sigma_d=0.2,
        sigma_a=15,
        angle_k=3
    )

    # 正向传播
    ref_feats_out, src_feats_out = geo_transformer(ref_points, src_points, ref_feats, src_feats)

    # 损失函数
    loss = (ref_feats_out.mean() + src_feats_out.mean())

    # 反向传播
    loss.backward()

    # print(ref_points.grad)
    # print(src_points.grad)
    # print(ref_feats.grad)
    # print(src_feats.grad)
    print(ref_feats_out.shape)
    print(src_feats_out.shape)