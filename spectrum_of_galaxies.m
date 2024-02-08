% Defining Neural Network class
classdef NN < handle
    properties
        fc1
        relu
        fc2
    end
    
    methods
        function obj = NN(input_size, hidden_size, output_size)
            obj.fc1 = randn(input_size, hidden_size);
            obj.relu = @(x) max(0, x);
            obj.fc2 = randn(hidden_size, output_size);
        end
        
        function output = forward(obj, x)
            x = x * obj.fc1;
            x = obj.relu(x);
            output = x * obj.fc2;
        end
    end
end

% Generating random data
data_spectra = rand(100, 10);
labels = randi([0, 1], 100, 1);

% Creating MATLAB Engine
eng = matlab.engine.start_matlab();

% Converting data to MATLAB data type
matlab_spectra = matlab.double(data_spectra);

% Processing spectra using MATLAB function
processed_results = eng.process_spectres(matlab_spectra);

% Converting processed results to Python compatible format
results_python = cell2mat(processed_results);

% Converting results to Torch tensor
inputs = torch.tensor(results_python, 'dtype', torch.float32);
targets = torch.tensor(labels, 'dtype', torch.int64);

% Splitting data into train and test sets
[train_inputs, test_inputs, train_targets, test_targets] = train_test_split(inputs, targets, 'test_size', 0.2, 'random_state', 42);

% Initializing neural network parameters
input_size = size(train_inputs, 2);
hidden_size = 64;
output_size = 2;

% Creating model instance
model = NN(input_size, hidden_size, output_size);

% Defining loss function and optimizer
criterion = nn.CrossEntropyLoss();
optimizer = optim.Adam(model.parameters(), 'lr', 0.001);

% Training the model
num_epochs = 10;
batch_size = 32;
for epoch = 1:num_epochs
    for i = 1:batch_size:length(train_inputs)
        inputs_batch = train_inputs(i:min(i+batch_size-1, end), :);
        targets_batch = train_targets(i:min(i+batch_size-1, end));
        
        optimizer.zero_grad();
        outputs = model.forward(inputs_batch);
        loss = criterion(outputs, targets_batch);
        loss.backward();
        optimizer.step();
    end
    fprintf('Epoch [%d/%d], Loss: %.4f\n', epoch, num_epochs, loss.item());
end

% Testing the model
outputs = model.forward(test_inputs);
[~, predicted] = max(outputs, [], 2);
accuracy = sum(predicted == test_targets) / length(test_targets);
fprintf('Model accuracy on test data: %.4f\n', accuracy);

% Calculating measures of central tendency
mean_val = mean(results_python);
median_val = median(results_python);
mode_val = mode(results_python);

fprintf('Mean: %.4f\n', mean_val);
fprintf('Median: %.4f\n', median_val);
fprintf('Mode: %.4f\n', mode_val);

% Plotting measures of central tendency
measures = containers.Map({'Mean', 'Median', 'Mode'}, [mean_val, median_val, mode_val]);
bar(measures.keys, cell2mat(measures.values));
title('Measures of Central Tendency');
ylabel('Values');
