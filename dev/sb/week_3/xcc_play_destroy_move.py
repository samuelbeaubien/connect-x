import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

# Define the ResNet-18 model for Connect-4
class ResNet18Connect4(nn.Module):
    def __init__(self, output_size):
        super(ResNet18Connect4, self).__init__()
        # Load pre-trained ResNet18
        self.resnet = models.resnet18(pretrained=True)
        
        # Modify the first layer to accept 7x9 input instead of 224x224
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Modify the final layer to output the number of possible moves (9 columns)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, output_size)
    
    def forward(self, x):
        return self.resnet(x)

# Example: Connect-4 has 9 columns, so 9 possible moves
output_size = 9  # 9 columns in Connect-4
model = ResNet18Connect4(output_size)

# Print model summary (to confirm)
print(model)

# Example of dummy input: Batch of 32, 1 channel (grayscale), 7x9 board states
inputs = torch.randn(32, 1, 7, 9)  # Batch size of 32
targets = torch.randint(0, 9, (32,))  # Target moves, randomly generated for now (actual data would be needed)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (simplified)
for epoch in range(10):  # 10 epochs for this example
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(inputs)
    
    # Compute loss
    loss = criterion(outputs, targets)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Example of making a prediction with the trained model
with torch.no_grad():
    test_board = torch.randn(1, 1, 7, 9)  # Example test input (one 7x9 board)
    predicted_move = model(test_board)
    move = predicted_move.argmax(dim=1)  # Get the column with the highest prediction
    print(f"Predicted move: {move.item()}")
