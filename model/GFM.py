
import torch
from thop import profile
from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange

from model.utils import LayerNorm2d

class SKFusion(nn.Module):
	def __init__(self, dim, height=2, reduction=8):
		super(SKFusion, self).__init__()

		self.height = height
		d = max(int(dim/reduction), 4)

		self.mlp = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(dim, d, 1, bias=False),
			nn.ReLU(True),
			nn.Conv2d(d, dim*height, 1, bias=False)
		)

		self.softmax = nn.Softmax(dim=1)

	def forward(self, in_feats):
		B, C, H, W = in_feats[0].shape

		in_feats = torch.cat(in_feats, dim=1)
		in_feats = in_feats.view(B, self.height, C, H, W)

		feats_sum = torch.sum(in_feats, dim=1)
		attn = self.mlp(feats_sum)
		attn = self.softmax(attn.view(B, self.height, C, 1, 1))

		out = torch.sum(in_feats*attn, dim=1)
		return out


"""Grafting Fusion Module"""
class GFM(nn.Module):
    def __init__(self,dim,num_heads):
        super(GFM, self).__init__()
        self.num_heads = num_heads
        self.T_norm = LayerNorm2d(dim)
        self.C_norm = LayerNorm2d(dim)
        """fusion T_E and C_E"""
        self.T_q_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.T_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.C_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.C_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.C_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.C_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

        self.SK_fusion = SKFusion(dim = dim)

    def forward(self,T_E_input,C_E_input,T_D_input):
        b, c, h, w = T_E_input.shape
        T_E = self.T_norm(T_E_input)
        C_E = self.C_norm(C_E_input)

        """fusion T_E and C_E"""
        T_E_q = self.T_q_dwconv(self.T_q_conv(T_E))
        T_C_k = self.C_k_dwconv(self.C_k_conv(C_E))
        T_C_v = self.C_v_dwconv(self.C_v_conv(C_E))

        T_E_q = rearrange(T_E_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        T_C_k = rearrange(T_C_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        T_C_v = rearrange(T_C_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        T_E_q = torch.nn.functional.normalize(T_E_q, dim=-1)
        T_C_k = torch.nn.functional.normalize(T_C_k, dim=-1)

        _,_,C,_ = T_E_q.shape
        """top_k mask"""
        mask1 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)

        attn = (T_E_q @ T_C_k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ T_C_v)
        out2 = (attn2 @ T_C_v)
        out3 = (attn3 @ T_C_v)
        out4 = (attn4 @ T_C_v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        fusion1 = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion1 = self.project_out(fusion1) + C_E_input

        """fusion E and decoder"""
        fusion2 = self.SK_fusion([fusion1,T_D_input])

        return fusion2

"""Cross Sparse Attention"""
class CSA(nn.Module):
    def __init__(self,dim,num_heads):
        super(CSA, self).__init__()
        self.num_heads = num_heads
        self.T_norm = LayerNorm2d(dim)
        self.C_norm = LayerNorm2d(dim)
        """fusion T_E and C_E"""
        self.T_q_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.T_q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.C_k_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.C_k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.C_v_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.C_v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)


    def forward(self,T_E_input,C_E_input):
        b, c, h, w = T_E_input.shape
        T_E = self.T_norm(T_E_input)
        C_E = self.C_norm(C_E_input)

        """fusion T_E and C_E"""
        T_E_q = self.T_q_dwconv(self.T_q_conv(T_E))
        T_C_k = self.C_k_dwconv(self.C_k_conv(C_E))
        T_C_v = self.C_v_dwconv(self.C_v_conv(C_E))

        T_E_q = rearrange(T_E_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        T_C_k = rearrange(T_C_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        T_C_v = rearrange(T_C_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        T_E_q = torch.nn.functional.normalize(T_E_q, dim=-1)
        T_C_k = torch.nn.functional.normalize(T_C_k, dim=-1)

        _,_,C,_ = T_E_q.shape
        """top_k mask"""
        mask1 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask2 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask3 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)
        mask4 = torch.zeros(b, self.num_heads, C, C, device=T_E_input.device, requires_grad=False)

        attn = (T_E_q @ T_C_k.transpose(-2, -1)) * self.temperature

        index = torch.topk(attn, k=int(C/2), dim=-1, largest=True)[1]
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*2/3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*3/4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C*4/5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ T_C_v)
        out2 = (attn2 @ T_C_v)
        out3 = (attn3 @ T_C_v)
        out4 = (attn4 @ T_C_v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        fusion = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        fusion = self.project_out(fusion) + C_E_input

        return fusion

if __name__ == '__main__':
	q = torch.randn((2, 32, 256, 256))
	k = torch.randn((2, 32, 256, 256))
	v = torch.randn((2, 32, 256, 256))
	net = GFM(dim = 32,num_heads=8)
	flops, params = profile(net, inputs=(q,k,v,))
	print(' Number of parameters:%.4f M' % (params / 1e6))
	print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))



