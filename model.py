# We need to host alot of models here and return them based on the argument 
# so it can be mix of functionl as well as class based script 
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double Convolution Block: (Conv2d -> BatchNorm -> ReLU) × 2
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downsampling block for U-Net: MaxPool followed by DoubleConv
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)

class Up(nn.Module):
    """
    Upsampling block for U-Net: Upsample followed by DoubleConv
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        bilinear (bool): Whether to use bilinear upsampling
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Ensure the dimensions match for concatenation
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final convolution layer to produce the output segmentation map
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    U-Net architecture for image segmentation tasks
    
    Args:
        n_channels (int): Number of input channels
        n_classes (int): Number of output classes (1 for binary segmentation)
        bilinear (bool): Whether to use bilinear upsampling
    """
    def __init__(self, checkpoint_path, in_channels=1, n_classes=2, bilinear=True):
        super(UNet, self).__init__()

        self.checkpoint_path = checkpoint_path
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoding path
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # Decoding path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Final output layer
        self.outc = OutConv(64, n_classes) # For thyroid + nodule make 2 outputs
        
        self._load_pretrained_weights()


    def _load_pretrained_weights(self):

        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))

        pretrained_state_dict = checkpoint['model_state_dict']
        current_state_dict = self.state_dict()

        compatible = {
            key: value
            for key, value in pretrained_state_dict.items()
            if key in current_state_dict
            and value.shape == current_state_dict[key].shape
        }

        missing, unexpected = self.load_state_dict(compatible, strict=False)

        print(f"Loaded {len(compatible)} pretrained tensors")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)


    def forward(self, x):

        if x.shape[1] != self.in_channels:
            x = x.repeat(1, self.in_channels, 1, 1)  # Repeat channels if input has fewer channels than expected

        # Encoding path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoding path with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final output
        x = self.outc(x)
        return x
