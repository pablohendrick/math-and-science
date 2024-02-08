import matlab.engine
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

eng = matlab.engine.start_matlab()

data_spectra = np.random.rand(100, 10)
labels = np.random.randint(0, 2, 100)

matlab_spectra = matlab.double(data_spectra.tolist())

processed_results = eng.process_spectres(matlab_spectra)

results_python = np.array(processed_results._data).flatten()

inputs = torch.tensor(results_python, dtype=torch.float32)
targets = torch.tensor(labels, dtype=torch.long)

train_inputs, test_inputs, train_targets, test_targets = train_test_split(inputs, targets, test_size=0.2, random_state=42)

input_size = train_inputs.shape[1]
hidden_size = 64
output_size = 2

model = NN(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(train_inputs), batch_size):
        inputs_batch = train_inputs[i:i+batch_size]
        targets_batch = train_targets[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs_batch)
        loss = criterion(outputs, targets_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    outputs = model(test_inputs)
    _, predicted = torch.max(outputs, 1)
    accuracy = (predicted == test_targets).sum().item() / len(test_targets)
    print(f'Model accuracy on test data: {accuracy:.4f}')
  
mean_val = np.mean(results_python)
median_val = np.median(results_python)
mode_val = float(torch.mode(torch.tensor(results_python))[0])

print(f'Mean: {mean_val:.4f}')
print(f'Median: {median_val:.4f}')
print(f'Mode: {mode_val:.4f}')

measures = {'Mean': mean_val, 'Median': median_val, 'Mode': mode_val}
plt.bar(measures.keys(), measures.values())
plt.title('Measures of Central Tendency')
plt.ylabel('Values')
plt.show()

eng.quit()
