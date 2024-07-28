import torch
import torch.nn as nn

class my_CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(my_CNN, self).__init__()
        
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
       
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.pool2 = nn.AvgPool2d(kernel_size=4)

        self.flatten = nn.Flatten()
        
        # Classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2560, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        
        a1 = self.pool1(x)
        a2 = self.pool2(x)
        
        f1 = self.flatten(a1)
        f2 = self.flatten(a2)
        
        concatenated_features = torch.cat((f1, f2), dim=1)
        
        output = self.classifier(concatenated_features)
        
        return output
