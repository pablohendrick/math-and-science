using CSV
using DataFrames
using Random
using Flux
using Flux: binarycrossentropy, params, throttle
using Flux.Optimise: ADAM
using Statistics: mean

# File names
file_general_chemistry = "general_chemistry.csv"
file_physicochemical = "physicochemical.csv"
file_organic_chemistry = "organic_chemistry.csv"
file_inorganic_chemistry = "inorganic_chemistry.csv"

# Reading CSV files into DataFrames
df_general_chemistry = CSV.read(file_general_chemistry)
df_physicochemical = CSV.read(file_physicochemical)
df_organic_chemistry = CSV.read(file_organic_chemistry)
df_inorganic_chemistry = CSV.read(file_inorganic_chemistry)

# DataFrames collection
dataframe = (df_general_chemistry, df_physicochemical, df_organic_chemistry, df_inorganic_chemistry)

# Variable attributes
variable_attributes = [
    :number_of_atoms,
    :molar_mass,
    :surface_area,
    :volume,
    :bond_length,
    :electronegativity,
    :polarity,
    :solubility,
    :formal_charge,
    :dipole_moment,
    :enthalpy,
    :ionization_energy
]

# Concatenating attributes for features (X) and labels (y)
X = vcat([df[variable_attributes] for df in dataframe]...)
y = vcat([df[variable_attributes] for df in dataframe]...)

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model initialization and training
model = LinearRegression()
fit!(model, X_train, y_train)

# Prediction using the trained model
y_pred = predict(model, X_test)

# Calculating Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

println("MSE (Mean Squared Error): ", mse)
println("R² (Coefficient of Determination): ", r2)

# Element, Molecule, Reaction, and Atom Number classes definition
mutable struct Element
    name::String
    symbol::String
    atomic_mass::Float64
    element_type::String
end

mutable struct Molecule
    name::String
    formula::String
    elements::Array{Element, 1}
end

mutable struct Reaction
    reactants::Array{Molecule, 1}
    products::Array{Molecule, 1}
end

mutable struct NumberOfAtoms
    element::Element
    quantity::Int64
end

# Organic and Inorganic Molecule classes as subclasses of Molecule
mutable struct Organic <: Molecule
    type::String
end

mutable struct Inorganic <: Molecule
    type::String
end

# Available elements and performed experiments
available_elements = Dict{String, Element}()
performed_experiments = []

# Function to add an element to available elements
function add_element(element::Element)
    available_elements[element.symbol] = element
end

# Function to list available elements
function list_elements()
    println("Elements available in the laboratory:")
    for (symbol, element) in available_elements
        println(element.name, " - Symbol: ", symbol, " - Atomic Mass: ", element.atomic_mass)
    end
end

# Function to conduct an experiment
function conduct_experiment(element_symbol::String, experiment_type::String, manual::Bool=true)
    global available_elements, performed_experiments
    if haskey(available_elements, element_symbol)
        element = available_elements[element_symbol]
        experiment_result = Dict("element" => element, "experiment_type" => experiment_type, "manual" => manual)
        println("Conducting experiment on ", element.name, " (", element.symbol, ") with experiment type: ", experiment_type, "...")
        push!(performed_experiments, experiment_result)
    else
        println("Element not found in the laboratory.")
    end
end

# Function to prepare data for the neural network
function prepare_data()
    global performed_experiments
    input_data = []
    output_data = []

    for experiment in performed_experiments
        input_data_point = [experiment["element"].atomic_mass, length(experiment["experiment_type"])]
        output_data_point = Int(experiment["manual"])

        push!(input_data, input_data_point)
        push!(output_data, output_data_point)
    end

    input_tensor = Flux.unsqueeze(torch.tensor(input_data, dtype=torch.float32), 2)
    output_tensor = torch.tensor(output_data, dtype=torch.float32)

    return input_tensor, output_tensor
end

# Neural Network model definition
mutable struct NeuralNetwork
    input_layer::Dense
    output_layer::Dense
end

NeuralNetwork(input_size::Int, hidden_size::Int) = NeuralNetwork(Dense(input_size, hidden_size, relu), Dense(hidden_size, 1, σ))

function (nn::NeuralNetwork)(x)
    x = nn.input_layer(x)
    x = nn.output_layer(x)
    return x
end

# Initializing chemical elements
element_h = Element("Hydrogen", "H", 1.008, "H")
element_o = Element("Oxygen", "O", 15.999, "gas")
element_fe = Element("Iron", "Fe", 55.845, "metal")
#...

add_element(element_h)
add_element(element_o)
add_element(element_fe)

list_elements()

elements = collect(keys(available_elements))
random_element = rand(elements)
experiment_types = ["Experiment A", "Experiment B", "Experiment C"]
random_experiment_type = rand(experiment_types)

conduct_experiment(random_element, random_experiment_type, false)

conduct_experiment("H", "Manual Experiment")

input_data, output_data = prepare_data()

model = NeuralNetwork(2, 5)
criterion = binarycrossentropy
optimizer = ADAM(params(model), 0.01)

for epoch in 1:1000
    Flux.train!(criterion, zip(input_data, output_data), optimizer)
    if epoch % 100 == 0
        loss = mean(binarycrossentropy.(model(input_data), output_data))
        println("Epoch ", epoch, "/", 1000, ", Loss: ", loss)
    end
end

test_example = Flux.unsqueeze(torch.tensor([[1.008, 15]], dtype=torch.float32), 2)
prediction = model(test_example)
println("Prediction for the test example: ", prediction)
