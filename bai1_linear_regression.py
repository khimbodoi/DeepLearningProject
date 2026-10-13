import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

class LinDataset(Dataset):
    def __init__(self, num_samp=100):
        self.x = torch.linspace(-1, 1, num_samp).unsqueeze(1)
        self.y = 2.435 * self.x + 1
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

dataset = LinDataset(num_samp=100)

loader = DataLoader(dataset, batch_size=10, shuffle=True)

class LinModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)
    def forward(self, x): 
        return self.linear(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = LinModel().to(device)

loss_fn = nn.MSELoss() 

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0 
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        prediction = model(x_batch)
        loss= loss_fn(prediction, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

with torch.no_grad():
    test_x = torch.tensor([[3.0]], device=device)
    prediction = model(test_x)
    print(f"Prediction for input 3: {prediction.item():.4f}")
        
