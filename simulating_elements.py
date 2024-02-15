import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import random

# File names
file_general_chemistry = 'general_chemistry.csv'
file_physicochemical = 'physicochemical.csv'
file_organic_chemistry = 'organic_chemistry.csv'
file_inorganic_chemistry = 'inorganic_chemistry.csv'

# Reading CSV files into DataFrames
df_general_chemistry = pd.read_csv(file_general_chemistry)
df_physicochemical = pd.read_csv(file_physicochemical)
df_organic_chemistry = pd.read_csv(file_organic_chemistry)
df_inorganic_chemistry = pd.read_csv(file_inorganic_chemistry)

# DataFrames collection
dataframe = (df_general_chemistry, df_physicochemical, df_organic_chemistry, df_inorganic_chemistry)
# Variable attributes
variable_attributes = [
    'number_of_atoms', 
    'molar_mass', 
    'surface_area', 
    'volume', 
    'bond_length',
    'electronegativity',
    'polarity',
    'solubility',
    'formal_charge',
    'dipole_moment',
    'enthalpy',
    'ionization_energy'
]

# Concatenating attributes for features (X) and labels (y)
X = pd.concat([df[variable_attributes] for df in dataframe], ignore_index=True)
y = pd.concat([df[variable_attributes] for df in dataframe], ignore_index=True)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model initialization and training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction using the trained model
y_pred = model.predict(X_test)

# Calculating Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE (Mean Squared Error): {mse}')
print(f'R² (Coefficient of Determination): {r2}')

# Element, Molecule, Reaction, and Atom Number classes definition
class Element:
    # Class for chemical elements
    def __init__(self, name, symbol, atomic_mass, element_type):
        self.name = name
        self.symbol = symbol
        self.atomic_mass = atomic_mass
        self.element_type = element_type 

class Molecule:
    # Class for molecules
    def __init__(self, name, formula, elements):
        self.name = name
        self.formula = formula
        self.elements = elements

class Reaction:
    # Class for chemical reactions
    def __init__(self, reactants, products):
        self.reactants = reactants 
        self.products = products 

class NumberOfAtoms:
    # Class to track the number of atoms of an element
    def __init__(self, element, quantity):
        self.element = element
        self.quantity = quantity 

# Organic and Inorganic Molecule classes as subclasses of Molecule
class Organic(Molecule):
    # Subclass representing organic molecules
    def __init__(self, name, formula, elements):
        super().__init__(name, formula, elements)
        self.type = "organic" 

class Inorganic(Molecule):
    # Subclass representing inorganic molecules
    def __init__(self, name, formula, elements):
        super().__init__(name, formula, elements)
        self.type = "inorganic" 

# Available elements and performed experiments
available_elements = {}
performed_experiments = []

# Function to add an element to available elements
def add_element(element):
    global available_elements
    available_elements[element.symbol] = element

# Function to list available elements
def list_elements():
    print("Elements available in the laboratory:")
    for symbol, element in available_elements.items():
        print(f"{element.name} - Symbol: {symbol} - Atomic Mass: {element.atomic_mass}")

# Function to conduct an experiment
def conduct_experiment(element_symbol, experiment_type, manual=True):
    global available_elements, performed_experiments
    if element_symbol in available_elements:
        element = available_elements[element_symbol]
        experiment_result = {
            "element": element,
            "experiment_type": experiment_type,
            "manual": manual
        }
        print(f"Conducting experiment on {element.name} ({element.symbol}) with experiment type: {experiment_type}...")
        
        performed_experiments.append(experiment_result)
    else:
        print("Element not found in the laboratory.")

# Function to prepare data for the neural network
def prepare_data():
    global performed_experiments
    input_data = []
    output_data = []

    for experiment in performed_experiments:
        input_data_point = [experiment['element'].atomic_mass, len(experiment['experiment_type'])]
        output_data_point = int(experiment['manual'])

        input_data.append(input_data_point)
        output_data.append(output_data_point)

    input_tensor = torch.tensor(input_data, dtype=torch.float32)
    output_tensor = torch.tensor(output_data, dtype=torch.float32)

    return input_tensor, output_tensor

# Neural Network class definition
class NeuralNetwork(nn.Module):
    # Neural Network model class
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.input_layer = nn.Linear(2, 5)
        self.output_layer = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.input_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x

# Initializing chemical elements
element_h = Element("Hydrogen", "H", 1.008, "H")
element_o = Element("Oxygen", "O", 15.999, "gas")
element_fe = Element("Iron", "Fe", 55.845, "metal")
#...

add_element(element_h)
add_element(element_o)
add_element(element_fe)

list_elements()

elements = list(available_elements.keys())
random_element = random.choice(elements)
experiment_types = ["Experiment A", "Experiment B", "Experiment C"]
random_experiment_type = random.choice(experiment_types)

conduct_experiment(random_element, random_experiment_type, manual=False)

conduct_experiment("H", "Manual Experiment")

input_data, output_data = prepare_data()

model = NeuralNetwork()

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    outputs = model(input_data)
    loss = criterion(outputs, output_data.view(-1, 1))
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}/{1000}, Loss: {loss.item()}')

test_example = torch.tensor([[1.008, 15]], dtype=torch.float32)
prediction = model(test_example)
print(f'Prediction for the test example: {prediction.item()}')
