"""
Thumbs Classification for Thumbs Up/Down using Transfer Learning from VGG
"""

import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from glob import glob
import time

# Preprocessing chain: resize, crop, tensor, normalize
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Build training and testing sets
test_set_label = 'Ahmad_Shour'
train_X, test_X = torch.zeros((0, 3, 224, 224)), torch.zeros((0, 3, 224, 224))
train_y, test_y = [], []

for sub in os.listdir('dataset'):
    print(f'Loading: {sub}')
    for category in ['down', 'up']:
        for file in glob(os.path.join('dataset', sub, category, '*.jpg')):
            input_image = Image.open(file)
            input_tensor = preprocess(input_image)
            if sub == test_set_label:
                test_X = torch.cat((test_X, input_tensor.unsqueeze(0)))
                if category == 'down':
                    test_y.append(-1.0)
                else:
                    test_y.append(1.0)
            else:
                train_X = torch.cat((train_X, input_tensor.unsqueeze(0)))
                if category == 'down':
                    train_y.append(-1.0)
                else:
                    train_y.append(1.0)

train_y = torch.tensor(train_y)
test_y = torch.tensor(test_y)


# Define a datatset class for the data
class LikeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx, :, :, :], self.y[idx]


# Construct data loader
train_loader = DataLoader(LikeDataset(train_X, train_y), batch_size=32, shuffle=True)
test_loader = DataLoader(LikeDataset(test_X, test_y), batch_size=32, shuffle=False)

# Load pretrained CNN
vgg16 = models.vgg16(pretrained=True)

# Freeze convolution weights
for param in vgg16.features.parameters():
    param.requires_grad = False

# Newly created modules have require_grad=True by default
num_features = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1]  # Remove last layer
features.extend([nn.Linear(num_features, 1)])  # Add our layer with one output
vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier

# Optimizer
optimizer = torch.optim.SGD(vgg16.classifier.parameters(), lr=0.001, momentum=0.9)
# Loss function
criterion = torch.nn.MSELoss()


# Training function
def fit(model, train_dataloader):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    for i, data in enumerate(train_dataloader):
        print(i)
        target = data[1].unsqueeze(1)
        data = data[0]
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        preds = 2 * (output > 0) - 1
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
    train_loss = train_running_loss / len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct / len(train_dataloader.dataset)
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')

    return train_loss, train_accuracy


# Training epochs
train_loss , train_accuracy = [], []
start = time.time()
for epoch in range(1):
    train_epoch_loss, train_epoch_accuracy = fit(vgg16, train_loader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)
end = time.time()
print((end - start) / 60, 'minutes')
