import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# Gerät konfigurieren (GPU oder CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyperparameter
num_epochs = 10
batch_size = 100
learning_rate = 0.001
# Datenlader für MNIST
transform = transforms.Compose([
   transforms.ToTensor(),
   transforms.Normalize((0.1307,), (0.3081,))  # MNIST Mean und Std
])
# Trainings- und Testdatensatz laden
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                        transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                       transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# CNN-Modell definieren
class ConvNet(nn.Module):
   def __init__(self):
       super(ConvNet, self).__init__()
       self.layer1 = nn.Sequential(
           nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),  # 1 Kanal (graustufig)
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2)
       )
       self.layer2 = nn.Sequential(
           nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2)
       )
       self.fc = nn.Linear(7*7*32, 10)  # 10 Klassen (0-9)
   def forward(self, x):
       out = self.layer1(x)
       out = self.layer2(out)
       out = out.reshape(out.size(0), -1)  # Flatten
       out = self.fc(out)
       return out
# Modell, Verlustfunktion und Optimierer initialisieren
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Trainingsschleife
total_steps = len(train_loader)
for epoch in range(num_epochs):
   model.train()
   for i, (images, labels) in enumerate(train_loader):
       images = images.to(device)
       labels = labels.to(device)
       # Forward Pass
       outputs = model(images)
       loss = criterion(outputs, labels)
       # Backward Pass und Optimierung
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
       if (i+1) % 100 == 0:
           print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')
print('Training abgeschlossen')
# Testschleife
model.eval()
with torch.no_grad():
   correct = 0
   total = 0
   for images, labels in test_loader:
       images = images.to(device)
       labels = labels.to(device)
       outputs = model(images)
       _, predicted = torch.max(outputs.data, 1)
       total += labels.size(0)
       correct += (predicted == labels).sum().item()
   print(f'Genauigkeit auf Testdaten: {100 * correct / total}%')
# Modell speichern (optional)
torch.save(model.state_dict(), 'mnist_cnn.pth')