import torch
import torch.nn as nn
from thop import profile
from model.restormer_arch import TransformerBlock, Upsample, Downsample
import torch.nn.functional as F
"""
"""
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

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class BasicLayer(nn.Module):
	def __init__(self, dim, depth, kernel_size):

		super().__init__()
		self.dim = dim
		self.depth = depth

		# build blocks
		self.blocks = nn.ModuleList([
			Refine_Block(dim, kernel_size=kernel_size)
			for i in range(depth)])

	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x

##---------- g_block (Simple Channel Spatial Attention Block) ----------
class Refine_Block(nn.Module):
    def __init__(self, dim, kernel_size = 1):
        super(Refine_Block, self).__init__()


        self.norm = LayerNorm2d(dim)
        self.input_conv = nn.Conv2d(dim , dim, kernel_size=1, bias=True)
        self.conv1x1 = nn.Conv2d(dim , dim, kernel_size=1, padding= 0, groups = 1, bias=True)
        self.conv3x3 = nn.Conv2d(dim , dim, kernel_size=3, padding= 1, groups = dim, bias=True)
        self.conv5x5 = nn.Conv2d(dim , dim, kernel_size=5, padding= 2, groups = dim, bias=True)
        self.gelu = nn.GELU()
        self.out_conv = nn.Conv2d(dim , dim, kernel_size=1, bias=True)


    def forward(self , x):
        fea = self.norm(x)
        fea = self.input_conv(fea)
        fea1 = self.conv1x1(fea)
        fea2 = self.conv3x3(fea)
        fea3 = self.conv5x5(fea)

        fea = fea1 * (fea2 + fea3)
        fea = self.gelu(fea)
        out = self.out_conv(fea)
        out += x
        return out





class network_v22(nn.Module):
    def __init__(self, ):
        super(network_v22, self).__init__()

        self.gUnet = Refine_Block(base_dim=32, depths=[1, 1, 3, 1, 1])
    def forward(self, inp):

        derain = self.gUnet(inp)
        return derain



if __name__ == '__main__':
	x = torch.randn((1, 3, 128, 128))
	net = network_v22()
	flops, params = profile(net, inputs=(x,))
	print(' Number of parameters:%.4f M' % (params / 1e6))
	print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))

