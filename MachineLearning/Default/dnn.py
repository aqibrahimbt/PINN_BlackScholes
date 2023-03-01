import torch
import torch.nn as nn
import torch.optim as optim

class BlackScholes(nn.Module):
    def __init__(self):
        super(BlackScholes, self).__init__()
        self.linear1 = nn.Linear(3, 32) # 3 input features
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 1) # 1 output feature

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# Initialize model
model = BlackScholes()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define training loop
def train(model, optimizer, criterion, epochs, train_loader):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('Epoch %d: loss=%.3f' % (epoch + 1, running_loss / len(train_loader)))