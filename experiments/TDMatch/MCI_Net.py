import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]

    return idx[:, :, :]


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class Feature_Embedding(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(Feature_Embedding, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )

    def forward(self, x):
        x1 = self.right(x)
        out = self.left(x)
        out = out + x1
        return torch.relu(out)



class ANA_Layer(nn.Module):
    def __init__(self, knn_num=9, in_channel=256):
        super(ANA_Layer, self).__init__()
        self.knn_num = knn_num
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel * 2, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )
        self.match_net = nn.Sequential(
            nn.Linear(self.in_channel * 2, self.in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channel // 2, 1)
        )
        self.weight = nn.Parameter(torch.zeros(1))
    def forward(self, x, identify):
        B, C, N, K = x.shape
        out = x
        match_input = x.permute(0, 2, 3, 1)
        S_ij = self.match_net(match_input).squeeze(-1)  # [B, N, K]
        S_ij = F.softmax(S_ij, dim=-1).unsqueeze(1)  # 归一化 [B, 1, N, K]

        # 计算加权邻域特征
        out = (out * S_ij).sum(-1, keepdim=True)  # [B, C, N]
        out = self.conv(out)
        # 加入 shortcut 连接
        out = identify + self.weight * out


        return out


class GCN(nn.Module):
    def __init__(self, in_channel):
        super(GCN, self).__init__()
        self.in_channel = in_channel
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.in_channel, (1, 1)),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
        )

    def gcn(self, x, w):
        B, _, N, _ = x.size()

        w = torch.relu(torch.tanh(w)).unsqueeze(-1)
        A = torch.bmm(w, w.transpose(1, 2))
        I = torch.eye(N).unsqueeze(0).to(x.device).detach()
        A = A + I
        D_out = torch.sum(A, dim=-1)
        D = (1 / D_out) ** 0.5
        D = torch.diag_embed(D)
        L = torch.bmm(D, A)
        L = torch.bmm(L, D)
        out = x.squeeze(-1).transpose(1, 2).contiguous()
        out = torch.bmm(L, out).unsqueeze(-1)
        out = out.transpose(1, 2).contiguous()

        return out

    def forward(self, x, w):
        out = self.gcn(x, w)
        out = self.conv(out)
        return out


class ICI_Layer(nn.Module):
    def __init__(self, in_channel):
        nn.Module.__init__(self)

        self.attq = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )
        self.attk = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // 4, kernel_size=1),
            nn.BatchNorm2d(in_channel // 4),
            nn.ReLU()
        )
        self.attv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, n1, n2, n3):
        q1 = self.attq(n1).squeeze(3)
        k1 = self.attk(n2).squeeze(3)
        v1 = self.attv(n3).squeeze(3)
        scores = torch.bmm(q1.transpose(1, 2), k1)
        att = torch.softmax(scores, dim=2)
        out = torch.bmm(v1, att.transpose(1, 2))
        out = out.unsqueeze(3)
        out = self.conv(out)
        out = n3 + self.gamma * out
        return out

class IFD_Layer(nn.Module):
    def __init__(self, channels, reduction=4):
        super(IFD_Layer, self).__init__()
        inter_channels = int(channels // reduction)

        # 自定义通道重排的LayerNorm层
        class ChannelLayerNorm(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.norm = nn.LayerNorm(dim)

            def forward(self, x):
                # 输入形状: [B, C, L]
                x = x.permute(0, 2, 1)  # 转换为 [B, L, C]
                x = self.norm(x)
                return x.permute(0, 2, 1)  # 恢复原形状 [B, C, L]

        self.cal_imp_score = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 1, 1)
        )
        self.conv = nn.Sequential(
            nn.Conv1d(channels, inter_channels, kernel_size=1),
            ChannelLayerNorm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(2 * channels, channels, kernel_size=1),
            ChannelLayerNorm(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, patch):
        x = patch
        patch = patch.squeeze(-1)  # B,D,N
        pred_importance = self.cal_imp_score(patch)
        w = pred_importance  # B*1*N
        w = torch.tanh(torch.relu(w))
        w = F.normalize(w, p=1, dim=2)

        patch_w = torch.mul(patch, w.expand_as(patch))  # B*D*N
        patch_sum = torch.sum(patch_w, dim=2, keepdim=True)  # B*D*1
        global_context = F.normalize(self.conv(patch_sum), p=2, dim=1)  # B*D*1

        proj_length = torch.bmm(patch.transpose(1, 2), global_context).transpose(1, 2)  # B*1*N
        proj = torch.mul(proj_length, global_context)  # B*D*N
        orth_comp = patch - proj  # B*D*N
        final_feat = patch + self.conv1(torch.cat([orth_comp, global_context.expand_as(orth_comp)], dim=1))  # B*D*N
        final_feat = final_feat.unsqueeze(-1)
        return final_feat


class MCI_Net(nn.Module):
    def __init__(self, out_channel=256, k_num=30):
        super(MCI_Net, self).__init__()
        self.in_channel = 256
        self.out_channel = out_channel
        self.k_num = k_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, (1, 1)),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU(inplace=True)
        )

        self.gcn1 = GCN(self.out_channel)

        self.ana2 = ANA_Layer(self.k_num, self.out_channel)
        self.ana3 = ANA_Layer(self.k_num, self.out_channel)

        self.ifd1 = IFD_Layer(self.out_channel)
        self.ifd2 = IFD_Layer(self.out_channel)

        self.ici1 = ICI_Layer(self.out_channel)
        self.ici2 = ICI_Layer(self.out_channel)
        self.ici3 = ICI_Layer(self.out_channel)

        self.proj = nn.Linear(self.out_channel * 3 , self.out_channel)

        self.embed_00 = nn.Sequential(
            Feature_Embedding(self.out_channel, self.out_channel, pre=False),
        )

        self.embed_01 = nn.Sequential(
            Feature_Embedding(self.out_channel * 3, self.out_channel * 3, pre=False),
        )


        self.mlp1 = nn.Conv2d(self.out_channel, 1, (1, 1))



    def forward(self, x):
        B, _, N, _ = x.size()
        x = self.proj(x)
        out = x.transpose(1, 3).contiguous() # 1 768 157 1 B N C 1
        # # B C N 1
        C_C1 = out

        out = self.conv(out)
        out = self.embed_00(out)
        knn_out = out.squeeze(-1).permute(0,2,1)
        k = min(self.k_num,knn_out.size(1))
        knn_out = out.squeeze(-1).permute(0, 2, 1)
        idx_fn1 = knn(out.squeeze(-1), k=k)
        if idx_fn1.size(-1) < self.k_num:  # 点云数量过少则补充
            pad_size = self.k_num - idx_fn1.size(-1)
            pad_indices = idx_fn1[:, :, 0].unsqueeze(-1).repeat(1, 1, pad_size)  # 复制第一个邻居点
            idx_fn1 = torch.cat([idx_fn1, pad_indices], dim=-1)  # 拼接补全
        w_p1 = self.mlp1(out).view(B, -1)
        out_gs1 = self.gcn1(out, w_p1)
        idx_gn1 = knn(out_gs1.squeeze(-1), k=k)
        if idx_gn1.size(-1) < self.k_num:
            pad_size = self.k_num - idx_gn1.size(-1)
            pad_indices = idx_gn1[:, :, 0].unsqueeze(-1).repeat(1, 1, pad_size)  # 复制第一个邻居点
            idx_gn1 = torch.cat([idx_gn1, pad_indices], dim=-1)  # 补全

        C_F1 = self.ana2(get_graph_feature(out, k=self.k_num, idx=idx_fn1),C_C1)
        C_G1 = self.ana3(get_graph_feature(out_gs1, k=self.k_num, idx=idx_gn1),C_C1)

        C_F1 = self.ifd1(C_F1)
        C_G1 = self.ifd2(C_G1)

        I_C1 = self.ici1(C_F1, C_G1, C_C1)
        I_F1 = self.ici2(C_G1, C_C1, C_F1)
        I_G1 = self.ici3(C_C1, C_F1, C_G1)

        out = self.embed_01(torch.cat((I_C1, I_F1, I_G1), 1))
        out = out.squeeze(3).permute(0,2,1).squeeze(0)
        return out


