# author:zxj
# date:2023/5/29 20:58

import torch
import torch.nn as nn
from thop import profile

from model.NAFBlock import NAFBlock
from model.GFM import GFM, CSA, SKFusion
from model.g_Unet import BasicLayer
from model.restormer_arch import TransformerBlock, Upsample, Downsample
import torch.nn.functional as F


class Restormer_E(nn.Module):
    def __init__(self,
                 dim=48,
                 num_blocks=[2, 2, 2, 2],
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        super(Restormer_E, self, ).__init__()

        self.conv_g_inp = nn.Conv2d(3, dim, kernel_size=3, padding = 1)

        self.encoder_T3 = nn.Sequential(*[
            TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                             LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.down3_4 = Downsample(dim)

        self.encoder_T4 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.down4_5 = Downsample(int(dim * 2 ** 1))

        self.encoder_T5 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.down5_6 = Downsample(int(dim * 2 ** 2))

        self.encoder_T6 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

    def forward(self, inp_img):
        inp_fea = self.conv_g_inp(inp_img)

        encoder_T3 = self.encoder_T3(inp_fea)
        down3_4 = self.down3_4(encoder_T3)

        encoder_T4 = self.encoder_T4(down3_4)
        down4_5 = self.down4_5(encoder_T4)

        encoder_T5 = self.encoder_T5(down4_5)
        down5_6 = self.down5_6(encoder_T5)

        encoder_T6 = self.encoder_T6(down5_6)

        return encoder_T3, encoder_T4, encoder_T5, encoder_T6



class Gunet_E(nn.Module):
    def __init__(self,
                 dim=32,
                 num_blocks=[1, 1, 1, 12],
                 ):
        super(Gunet_E, self, ).__init__()

        self.conv_inp = nn.Conv2d(3, dim, kernel_size=3, padding = 1)

        self.encoder_C1 = nn.Sequential(*[NAFBlock(c = int(dim * 2 ** 0)) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(int(dim * 2 ** 0))

        self.encoder_C2 = nn.Sequential(*[NAFBlock(c = int(dim * 2 ** 1)) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_C3 = nn.Sequential(*[NAFBlock(c = int(dim * 2 ** 2)) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 2 ** 2))

        self.encoder_C4 = nn.Sequential(*[NAFBlock(c = int(dim * 2 ** 3)) for i in range(num_blocks[3])])

    def forward(self, inp_img):
        inp_fea = self.conv_inp(inp_img)

        encoder_C1 = self.encoder_C1(inp_fea)
        down1_2 = self.down1_2(encoder_C1)

        encoder_C2 = self.encoder_C2(down1_2)
        down2_3 = self.down2_3(encoder_C2)

        encoder_C3 = self.encoder_C3(down2_3)
        down3_4 = self.down3_4(encoder_C3)

        encoder_C4 = self.encoder_C4(down3_4)

        return encoder_C1, encoder_C2, encoder_C3, encoder_C4



class Decoder(nn.Module):
    def __init__(self,
                 T_dim = 48,
                 C_dim = 32,
                 num_blocks = [2, 2, 14, 1, 1, 1],
                 heads = [8,4,2,1],         #这边可以重新设置
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 ):
        super(Decoder, self, ).__init__()

        self.decoder_T6 = nn.Sequential(*[
            TransformerBlock(dim=int(T_dim * 2 ** 3), num_heads = heads[0], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.up6_5 = Upsample(int(T_dim * 2 ** 3))
        self.fusion5 = SKFusion(dim = int(T_dim * 2 ** 2))
        self.decoder_T5 = nn.Sequential(*[
            TransformerBlock(dim=int(T_dim * 2 ** 2), num_heads = heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up5_4 = Upsample(int(T_dim * 2 ** 2))
        self.conv_T4=nn.Sequential(
            nn.Conv2d(T_dim * 2 ** 1, C_dim * 2 ** 3, kernel_size=3,stride=1,dilation=1,padding=1),
            nn.BatchNorm2d(C_dim * 2 ** 3),
            nn.ReLU(inplace=True)
            )
        self.conv_up5_4 = nn.Sequential(
            nn.Conv2d(T_dim * 2 ** 1, C_dim * 2 ** 3, kernel_size=3,stride=1,dilation=1,padding=1),
            nn.BatchNorm2d(C_dim * 2 ** 3),
            nn.ReLU(inplace=True)
            )
        self.fusion4 = GFM(dim = C_dim * 2 ** 3,num_heads = heads[2])
        self.decoder_C4 = nn.Sequential(*[NAFBlock(c = int(C_dim * 2 ** 3)) for i in range(num_blocks[2])])

        self.up4_3 = Upsample(int(C_dim * 2 ** 3))
        self.conv_T3=nn.Sequential(
            nn.Conv2d(T_dim * 2 ** 0, C_dim * 2 ** 2, kernel_size=3,stride=1,dilation=1,padding=1),
            nn.BatchNorm2d(C_dim * 2 ** 2),
            nn.ReLU(inplace=True)
            )
        self.fusion3 = GFM(dim = C_dim * 2 ** 2,num_heads = heads[3])
        self.decoder_C3 = nn.Sequential(*[NAFBlock(c = int(C_dim * 2 ** 2)) for i in range(num_blocks[3])])

        self.up3_2 = Upsample(int(C_dim * 2 ** 2))
        self.fusion2 = SKFusion(dim = int(C_dim * 2 ** 1))
        self.decoder_C2 = nn.Sequential(*[NAFBlock(c = int(C_dim * 2 ** 1)) for i in range(num_blocks[4])])

        self.up2_1 = Upsample(int(C_dim * 2 ** 1))
        self.fusion1 = SKFusion(dim = int(C_dim * 2 ** 0))
        self.decoder_C1 = nn.Sequential(*[NAFBlock(c = int(C_dim * 2 ** 0)) for i in range(num_blocks[5])])


    def forward(self, T3, T4, T5, T6, C1, C2, C3, C4):

        D6 = self.decoder_T6(T6)

        up6_5 = self.up6_5(D6)
        fusion5 = self.fusion5([up6_5,T5])
        D5 = self.decoder_T5(fusion5)

        up5_4 = self.up5_4(D5)
        up5_4 = self.conv_up5_4(up5_4)
        T4_ = self.conv_T4(T4)
        fusion4 = self.fusion4(T4_,C4,up5_4)
        D4 = self.decoder_C4(fusion4)

        up4_3 = self.up4_3(D4)
        T3_ = self.conv_T3(T3)
        fusion3 = self.fusion3(T3_,C3,up4_3)
        D3 = self.decoder_C3(fusion3)

        up3_2 = self.up3_2(D3)
        fusion2 = self.fusion2([up3_2,C2])
        D2 = self.decoder_C2(fusion2)

        up2_1 = self.up2_1(D2)
        fusion1 = self.fusion1([up2_1,C1])
        D1 = self.decoder_C1(fusion1)
        return D3,D1


class Refinement(nn.Module):
    def __init__(self,  base_dim=32, depths=[4, 4, 4, 4, 4, 4, 4], fusion_layer=SKFusion, kernel_size=[3, 3, 3, 1, 3, 3, 3]):
        super(Refinement, self).__init__()
        # setting
        assert len(depths) % 2 == 1
        stage_num = len(depths)
        half_num = stage_num // 2
        net_depth = sum(depths)
        embed_dims = [2 ** i * base_dim for i in range(half_num)]
        embed_dims = embed_dims + [2 ** half_num * base_dim] + embed_dims[::-1]

        self.patch_size = 2 ** (stage_num // 2)
        self.stage_num = stage_num
        self.half_num = half_num


        # backbone
        self.layers = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.fusions = nn.ModuleList()

        self.inp_conv = nn.Conv2d(6,base_dim,3,1,1)
        for i in range(self.stage_num):
            self.layers.append(
                BasicLayer(dim=embed_dims[i], depth=depths[i], kernel_size=kernel_size[i]))


        for i in range(self.half_num):

            self.downs.append(Downsample(embed_dims[i]))
            self.ups.append(Upsample(embed_dims[i + 1]))
            self.skips.append(nn.Conv2d(embed_dims[i], embed_dims[i], 1))
            self.fusions.append(fusion_layer(embed_dims[i]))
        self.fusions.append(fusion_layer(embed_dims[self.half_num]))

        self.four_up = nn.Sequential(Upsample(embed_dims[2]),Upsample(embed_dims[1]))
        self.inp_fusion = CSA(dim = base_dim,num_heads = 1)
        # output convolution
        self.outconv = nn.Conv2d(base_dim , 3 , 3, 1,1)

    def forward(self, g_rain_fea,rain_fea,inp):


        g_fea = self.four_up(g_rain_fea)
        rain_fusion = self.inp_fusion(g_fea,rain_fea)
        feat = rain_fusion
        skips = []

        for i in range(self.half_num):
            feat = self.layers[i](feat)
            skips.append(self.skips[i](feat))
            feat = self.downs[i](feat)

        feat = self.layers[self.half_num](feat)

        for i in range(self.half_num -1, -1, -1):
            feat = self.ups[i](feat)
            feat = self.fusions[i]([feat, skips[i]])
            feat = self.layers[self.stage_num - i - 1](feat)

        refine = self.outconv(feat) + inp

        return refine


class Baseline(nn.Module):
    def __init__(self, ):
        super(Baseline, self).__init__()
        self.T_dim = 48
        self.C_dim = 32
        # self.C_dim = 24
        self.Restormer_E = Restormer_E(dim = self.T_dim)
        self.Gunet_E = Gunet_E(dim = self.C_dim)
        self.Decoder = Decoder(T_dim = self.T_dim, C_dim = self.C_dim)
        self.g_output = nn.Conv2d(self.C_dim * 2 ** 2, 3, 3, 1, 1)
        self.output = nn.Conv2d(self.C_dim, 3, 3, 1, 1)
        self.refine = Refinement(base_dim=32, depths=[1, 1, 1, 8, 1, 1, 1])


    def forward(self, inp):
        C1, C2, C3, C4 = self.Gunet_E(inp)

        g_inp = F.interpolate(inp, size=(inp.shape[2] // 4, inp.shape[3] // 4), mode='bilinear', align_corners=True)

        T3, T4, T5, T6 = self.Restormer_E(g_inp)

        D3,D1 = self.Decoder(T3, T4, T5, T6, C1, C2, C3, C4)

        g_derain = g_inp + self.g_output(D3)

        derain = inp + self.output(D1)

        refine = self.refine(D3,D1,inp)

        return g_derain,derain,refine

if __name__ == '__main__':
	x = torch.randn((1, 3, 256, 256))
	net = Baseline()
	flops, params = profile(net, inputs=(x,))
	print(' Number of parameters:%.4f M' % (params / 1e6))
	print(' Number of FLOPs:%.4f GFLOPs' % (flops / 1e9))