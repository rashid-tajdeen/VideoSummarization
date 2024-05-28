import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


# Custom neural network model for multi-label classification
class MultiLabelModel(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(MultiLabelModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Adjust the input size based on your image dimensions
        self.fc1 = nn.Linear(128 * (240//2) * (240//2), 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class CustomMultiLabelModel(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(CustomMultiLabelModel, self).__init__()
        self.conv1 = nn.Conv3d(num_channels, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool = nn.MaxPool3d(kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0))
        self.fc1 = nn.Linear(128 * 30 * 30 * 5, 512)  # Adjust the input size based on your 3D image dimensions
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute((0, 2, 1, 3, 4))

        print(x.size())
        x = self.conv1(x)
        x = self.relu(x)
        print("Shape after conv1:", x.shape)

        x = self.conv2(x)
        x = self.relu(x)
        print("Shape after conv2:", x.shape)

        x = self.pool(x)
        print("Shape after pooling:", x.shape)
        x = x.view(-1, 128 * 30 * 30 * 5)  # Adjust this line for your 3D image dimensions
        x = F.relu(self.fc1(x))
        print("Shape after fc1:", x.shape)
        x = self.fc2(x)
        print("Shape after fc2:", x.shape)
        return torch.sigmoid(x)


class VideoClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(VideoClassificationModel, self).__init__()
        # Load a pre-trained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)
        # Modify the final fully connected layer for your classification task
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.permute((0, 2, 1, 3, 4))
        return self.resnet(x)


class TModel2(nn.Module):
    def __init__(self, num_frames=5, num_classes=1):
        super().__init__()
        output_dim = 512
        model = models.resnet18(pretrained=True)
        self.cls_model = nn.Sequential(*list(model.children())[:-1])
        self.classifier = nn.Sequential(nn.Linear(output_dim*num_frames, 1024, bias=True),
                                        nn.Hardswish(),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(1024, num_classes, bias=True))
        #self.num_images = num_images

        # Define learnable parameters matching the number of methods
        # Order corresponds to ['less_blur', 'motion', "histogram", "k_means"]
        self.method_priorities = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(3)
        ])

    def forward(self, img):
        B, T, C, H, W = img.size()
        img = img.view(B*T, C, H, W)
        out = self.cls_model(img)
        out = out.squeeze(2).squeeze(2)
        out = out.view(B, -1)

        # Incorporate the learnable parameters as additional biases
        params_sum = sum(self.method_priorities)
        out = out + params_sum

        out = self.classifier(out)
        return out


class Custom3DModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Custom3DModel, self).__init__()

        # Expected input shape: (num_of_videos, num_frames, channels, height, width) : (num_of_videos, 5, 3, 240, 240)

        # Define a 3D convolutional layer with 5 input channels
        self.conv1 = nn.Conv3d(input_shape[1], 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Add more convolutional layers and adjust architecture as needed
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)

        # Expected shape before flattening: (num_of_videos, 256, 3, 30, 30)
        self.flatten = nn.Flatten()
        # Expected shape after flattening: (num_of_videos, 691200)

        nodes_after_flattening = 256 * input_shape[2] * input_shape[3]/(2*2*2) * input_shape[4]/(2*2*2)  # 3 times pool
        self.fc1 = nn.Linear(int(nodes_after_flattening), 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Use Sigmoid activation for multi-label classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.sigmoid(x)

        return x


def main():
    model = Custom3DModel((100, 5, 3, 240, 240), 1)
    print(model)


if __name__ == '__main__':
    main()
