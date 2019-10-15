from torch import nn
from torchsummary import summary
from torchvision import models

from config import device


class MobileNetv2(nn.Module):
    def __init__(self):
        super(MobileNetv2, self).__init__()
        net = models.mobilenet_v2(pretrained=True)
        # Remove linear layer
        modules = list(net.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(1280, 512)

    def forward(self, images):
        x = self.backbone(images)  # [N, 1280, 7, 7]
        x = self.avg_pool(x)  # [N, 1280, 1, 1]
        x = x.view(-1, 1280)  # [N, 2048]
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = MobileNetv2().to(device)
    summary(model, input_size=(3, 112, 112))
