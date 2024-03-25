import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleAdaptation(nn.Module):
    def __init__(self, target_size=(4, 4)):
        super(StyleAdaptation, self).__init__()
        self.target_size = target_size
    
    def forward(self, style_features):
        # Use adaptive average pooling to resize the feature maps
        pooled_features = F.adaptive_avg_pool2d(style_features, self.target_size)
        return pooled_features

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * np.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.BatchNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        
        style = torch.flatten(style, 1)
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class CvBlock(nn.Module):
    '''(Conv2d => BN => ReLU) x 2'''
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class EncoderBlock(nn.Module):
    '''Pooling => (Conv2d => BN => ReLU)*2'''
    def __init__(self, in_ch, out_ch):
        super(EncoderBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            CvBlock(in_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)

class DecoderBlock(nn.Module):
	'''(Conv2d => BN => ReLU)*2 + Upscale'''
	def __init__(self, in_ch, out_ch):
		super(DecoderBlock, self).__init__()
		self.convblock = nn.Sequential(
			CvBlock(in_ch, in_ch),
			nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0)
		)

	def forward(self, x):
		return self.convblock(x)



class DenBlock(nn.Module):
    """Modified DenBlock to accept 2 input frames for the FastDVDnet model."""
    def __init__(self):
        super(DenBlock, self).__init__()

        self.chs_lyr = [32, 64, 128]

        # Adjusting InputCvBlock to accept 2 frames concatenated along the channel dimension
        self.inc = CvBlock(in_ch=2, out_ch=self.chs_lyr[0])
        self.enc0 = EncoderBlock(in_ch=self.chs_lyr[0], out_ch=self.chs_lyr[1])
        self.enc1 = EncoderBlock(in_ch=self.chs_lyr[1], out_ch=self.chs_lyr[2])
        self.dec2 = DecoderBlock(in_ch=self.chs_lyr[2], out_ch=self.chs_lyr[1])
        self.dec1 = DecoderBlock(in_ch=self.chs_lyr[1], out_ch=self.chs_lyr[0])
        self.outc = CvBlock(in_ch=self.chs_lyr[0], out_ch=self.chs_lyr[0])
        self.finalc = nn.Conv2d(self.chs_lyr[0], 1, kernel_size=3, padding=1, bias=False)

        self.adain_inc = AdaptiveInstanceNorm(self.chs_lyr[0], 4096)
        self.adain_0_e = AdaptiveInstanceNorm(self.chs_lyr[1], 4096)
        self.adain_1_e = AdaptiveInstanceNorm(self.chs_lyr[2], 4096)
        self.adain_2_d = AdaptiveInstanceNorm(self.chs_lyr[1], 4096)
        self.adain_1_d = AdaptiveInstanceNorm(self.chs_lyr[0], 4096)

    def forward(self, x, prior):

        x0 = self.inc(x)
        tmp = self.adain_inc(x0, prior)
        x1 = self.enc0(tmp)
        tmp = self.adain_0_e(x1, prior)
        x2 = self.enc1(tmp)
        tmp = self.adain_1_e(x2, prior)
        x2 = self.dec2(tmp)
        tmp = self.adain_2_d(x2, prior)
        x1 = self.dec1(x1 + x2)
        tmp = self.adain_1_d(x1, prior)
        x = self.outc(x0 + x1)
        x = self.finalc(x)

        return x

class AdaptiveFastDVDnet(nn.Module):

    def __init__(self):
        super(AdaptiveFastDVDnet, self).__init__()

        self.shared_den_block = DenBlock()
        self.final_den_block = DenBlock()

    def forward(self, x, prior):

        x0, x1, x2, x3 = [x[:, i, :, :].unsqueeze(1) for i in range(4)]

        input_02 = torch.cat([x0, x2], dim=1)  # Now shape [N, 2, H, W]
        input_13 = torch.cat([x1, x3], dim=1)  # Now shape [N, 2, H, W]

        # Process through the shared DenBlocks
        output_02 = self.shared_den_block(input_02, prior)
        output_13 = self.shared_den_block(input_13, prior)

        # Process the outputs through the final DenBlock
        final_input = torch.cat([output_02, output_13], dim=1)
        final_output = self.final_den_block(final_input, prior)

        return final_output
