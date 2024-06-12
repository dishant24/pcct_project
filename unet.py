import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchsummary import summary


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DoubleConv,self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
    def forward(self,x):
        return self.double_conv(x)
    
class UNET(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=1,
                 features=[64,128,256,512]):
        super(UNET,self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2)
            )
            self.ups.append(DoubleConv(feature*2,feature))

        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)

    def forward(self,x):
        skip_connection = []
        for down in self.downs:
            x = down(x)
            skip_connection.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connection = skip_connection[::-1]
        for i in range(0,len(self.ups),2):
            x = self.ups[i](x)
            skip = skip_connection[i//2]
            if x.shape != skip.shape:
                x = torchvision.transforms.functional.resize(x,size=skip.shape[2:])
            concat_skip = torch.cat((skip,x),dim=1)
            x = self.ups[i+1](concat_skip)

        return self.final_conv(x)
    
def test():
    x = torch.randn((3,1,161,161))
    model = UNET(in_channels=1,out_channels=1)
    summary(model, x.shape[1:])
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()

