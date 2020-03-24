import torch

try:
    from .DMFNet_16x import *
except:
    from DMFNet_16x import *


class AttentionBlock(nn.Module):
    def __init__(self, f_g, f_l, f_int, norm, g=1):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(f_g, f_int, kernel_size=1, padding=0, stride=1, groups=g,
                      bias=True),
            normalization(f_int, norm=norm)
            # nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(f_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(f_l, f_int, kernel_size=1, stride=1, padding=0, groups=g,
                      bias=True),
            normalization(f_int, norm=norm)
            # nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(f_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(f_int, 1, kernel_size=1, stride=1, padding=0,
                      bias=True),
            normalization(1, norm=norm),
            # nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttentionDMFNet(DMFNet):

    def __init__(self, c=4, n=32, channels=128, groups=16, norm='bn', num_classes=4):
        super(AttentionDMFNet, self).__init__(c, n, channels, groups, norm, num_classes)

        self.att3 = AttentionBlock(f_g=channels * 2, f_l=channels * 2, f_int=channels, norm=norm, g=16)
        self.att2 = AttentionBlock(f_g=channels * 2, f_l=channels, f_int=channels, norm=norm, g=16)
        self.att1 = AttentionBlock(f_g=channels, f_l=channels // 4, f_int=channels, norm=norm, g=16)

    def forward(self, x):
        # Encoder
        x1 = self.encoder_block1(x)  # H//2 down
        x2 = self.encoder_block2(x1)  # H//4 down
        x3 = self.encoder_block3(x2)  # H//8 down
        x4 = self.encoder_block4(x3)  # H//16
        # Decoder
        y1 = self.upsample1(x4)  # H//8
        x3 = self.att3(g=y1, x=x3)
        y1 = torch.cat([x3, y1], dim=1)
        y1 = self.decoder_block1(y1)

        y2 = self.upsample2(y1)  # H//4
        x2 = self.att2(g=y2, x=x2)
        y2 = torch.cat([x2, y2], dim=1)
        y2 = self.decoder_block2(y2)

        y3 = self.upsample3(y2)  # H//2
        x1 = self.att1(g=y3, x=x1)
        y3 = torch.cat([x1, y3], dim=1)
        y3 = self.decoder_block3(y3)

        y4 = self.upsample4(y3)
        y4 = self.seg(y4)
        if hasattr(self, 'softmax'):
            y4 = self.softmax(y4)
        return y4


if __name__ == '__main__':
    import os

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    x = torch.rand((1, 4, 128, 128, 128), device=device)  # [bsize,channels,Height,Width,Depth]
    model = AttentionDMFNet(c=4, groups=16, norm='sync_bn', num_classes=4)
    model.cuda(device)
    y = model(x)
    print(y.shape)
