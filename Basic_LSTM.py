import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends.cudnn import benchmark
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from random import randint
import optuna
from optuna.trial import TrialState






def load_and_prepare_data(df_CP_5):
    df = df_CP_5.copy()

    # Funzione per convertire la stringa in lista
    def parse_series(s):
        return np.array(ast.literal_eval(s), dtype=np.float32)

    # Applichiamo la funzione su tutte e tre le colonne
    features = ['heart_rate_data', 'respiration_data', 'stress_data']

    for col in features:
        df[col] = df[col].apply(parse_series)

    # Ora creiamo un array numpy shape (num_samples, 160, 3)
    X = np.stack(df[features].apply(lambda row: np.stack(row, axis=-1), axis=1).values)

    # Target
    y = df['Target_Unbiased'].astype(np.float32).values
    return X, y




"""
Data structure specifics:
- We have 34 sets of multivariate time series (3 dimensions)
- Each time series is 160 samples long
- Each set of 3 time series is associated with 1 ground truth value
- Input shape: X will be (34, 160, 3)
- Output shape: y will be (34,)
"""


# ------------------ PART 1: Custom Dataset Class ------------------

class MultivariateTimeSeriesDataset(Dataset):
    """
    Custom Dataset class for handling multivariate time series data.

    Args:
        X (numpy.ndarray): Input features with shape (num_samples, sequence_length, num_features)
                          In our case (num_series, 160, 3)
        y (numpy.ndarray): Target values with shape (num_samples,)
                          In our case (num_series,)
    """

    def __init__(self, X, y):
        # Convert numpy arrays to PyTorch tensors with float32 dtype
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ------------------ PART 2: LSTM Model Architecture ------------------

class LSTMRegressor(nn.Module):
    """
    LSTM model for regression on multivariate time series data.

    Args:
        input_size (int): Number of input features (3 in your case)
        hidden_size (int): Number of features in the hidden state of LSTM
        num_layers (int): Number of recurrent layers (stacked LSTM layers)
        dropout (float): Dropout probability for regularization
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMRegressor, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # (batch_size, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0  # Dropout between stacked LSTM layers
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, 1)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
                             In our case (batch_size, 160, 3)

        Returns:
            torch.Tensor: Model output of shape (batch_size, 1)
        """
        # Get batch size for reshaping hidden and cell states if needed
        batch_size = x.size(0)

        # LSTM output
        # output shape: (batch_size, seq_len, hidden_size)
        # hidden shape: (num_layers, batch_size, hidden_size)
        # cell shape: (num_layers, batch_size, hidden_size)
        output, (hidden, cell) = self.lstm(x)

        # There are two approaches we can take here:

        # Approach 1: Use the hidden state from the last layer
        # We take the hidden state from the last layer which contains
        # information from the entire sequence
        last_hidden = hidden[-1, :, :]  # Shape: (batch_size, hidden_size)

        # Apply dropout for regularization
        last_hidden = self.dropout(last_hidden)

        # Apply the fully connected layer to get the prediction
        out = self.fc(last_hidden)  # Shape: (batch_size, 1)

        return out.squeeze(1)  # Remove the extra dimension to get shape (batch_size,)


# ------------------ PART 3: Data Preprocessing Function ------------------

def preprocess_data(X, y, test_size=0.2, seed=42):
    """
    Preprocess the multivariate time series data, by standardizing the features and splitting into train/test sets.

    Args:
        X (numpy.ndarray): Input features of shape (num_series, sequence_length, num_features)
                          In our case (34, 160, 3)
        y (numpy.ndarray): Target values of shape (num_series,)
                          In our case (34,)
        test_size (float): Proportion of data to use for testing

    Returns:
        tuple: Preprocessed data as (X_train, X_test, y_train, y_test) ready for the model
    """
    # Note: In this case, we don't need to create sequences as each sample is already a full sequence

    # Standardize features - we need to reshape to 2D for StandardScaler, with first dimension as samples and second as features
    num_series, seq_length, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)  # Reshape to (num_series * seq_length, num_features), -1 means infer the size

    scaler_X = StandardScaler()
    X_scaled_reshaped = scaler_X.fit_transform(X_reshaped)

    # Reshape back to original shape
    X_scaled = X_scaled_reshaped.reshape(num_series, seq_length, num_features)

    # Standardize target
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten() # Reshape to 2D for StandardScaler, as a matrix with a shape of (num_series,1) and flatten back to 1D

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=seed
    )



    return X_train, X_test, y_train, y_test, scaler_X, scaler_y


# ------------------ PART 4: Training Function ------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    """
    Train the LSTM model.

    Args:
        model (nn.Module): LSTM model
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs (int): Maximum number of training epochs
        patience (int): Number of epochs to wait for validation loss improvement before early stopping

    Returns:
        tuple: Trained model and training history
    """
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Move model to device
    model = model.to(device)

    # Lists to store losses
    train_losses = []
    val_losses = []

    # Variables for early stopping
    best_val_loss = float('inf')
    counter = 0     #used to count the number of epochs without improvement, if the counter exceeds patience, training stops

    for epoch in range(num_epochs):
        #iterate for num_epochs times, each time training the model on the training set and validating it on the validation set
        #if the validation loss does not improve for patience epochs, training stops early

        # Training phase
        model.train()   #set the model to training mode setting layers that behave differently during training and evaluation (e.g., dropout, batch normalization) to training mode
        train_loss = 0.0

        for inputs, targets in train_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device) #set the inputs and targets to the same device as the model, save a copy of the data on the GPU if available

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate batch loss
            train_loss += loss.item() * inputs.size(0)  #inputs.size(0) is the batch size so that the train loss is weighted by the number of samples in the batch

        # Calculate average training loss for the epoch
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():  # No need to track gradients for validation
            for inputs, targets in val_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Calculate loss
                loss = criterion(outputs, targets)

                # Accumulate batch loss
                val_loss += loss.item() * inputs.size(0)

        # Calculate average validation loss for the epoch
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_lstm_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Load the best model
    model.load_state_dict(torch.load('best_lstm_model.pth'))

    return model, {"train_losses": train_losses, "val_losses": val_losses}


# ------------------ PART 5: Evaluation Function ------------------

def evaluate_model(model, test_loader, scaler_y):
    """
    Evaluate the trained model on the test data.

    Args:
        model (nn.Module): Trained LSTM model
        test_loader (DataLoader): DataLoader for test data
        scaler_y: StandardScaler used for scaling the target values

    Returns:
        tuple: Metrics and predictions
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            # Store predictions and actual values
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())

    # Convert to numpy arrays
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Inverse transform predictions and actual values
    predictions = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
    benchmark = np.zeros_like(predictions)

    # Calculate metrics
    test_mse = np.mean((predictions - actuals) ** 2)
    test_rmse = np.sqrt(test_mse)
    test_mae = np.mean(np.abs(predictions - actuals))

    # Calculate benchmark metrics
    benchmark_mse = np.mean((benchmark - actuals) ** 2)
    benchmark_rmse = np.sqrt(benchmark_mse)
    benchmark_mae = np.mean(np.abs(benchmark - actuals))

    print("== Metriche modello ==")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    print("\n\n== Benchmark (predice 0) ==")
    print(f"Benchmark MSE: {benchmark_mse:.4f}")
    print(f"Benchmark RMSE: {benchmark_rmse:.4f}")
    print(f"Benchmark MAE: {benchmark_mae:.4f}")

    return {
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "benchmark_mse": benchmark_mse,
        "benchmark_rmse": benchmark_rmse,
        "benchmark_mae": benchmark_mae,
        "predictions": predictions,
        "actuals": actuals
    }


# ------------------ PART 6: Visualization Function ------------------

def plot_results(history, evaluation_results):
    """
    Plot training/validation loss and predictions vs actual values.

    Args:
        history (dict): Training history containing losses
        evaluation_results (dict): Evaluation metrics and predictions
    """
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot training and validation loss
    ax1.plot(history["train_losses"], label="Training Loss")
    ax1.plot(history["val_losses"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()

    # Plot predictions vs actual values
    predictions = evaluation_results["predictions"]
    actuals = evaluation_results["actuals"]
    benchmark = np.zeros_like(predictions)

    # Scatter actual vs predicted
    ax2.scatter(actuals, predictions, alpha=0.5, label="Predictions")

    # Line of equality
    ax2.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--', label="Equality line")

    # Benchmark line: predicted = 0, quindi x=actuals, y=0
    ax2.plot(actuals, benchmark, linestyle="--", color="gray",
             label=f"Benchmark (RMSE: {evaluation_results['benchmark_rmse']:.4f})")

    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Predicted Values")
    ax2.set_title(f"Predictions vs Actual Values (Model RMSE: {evaluation_results['test_rmse']:.4f})")

    ax2.legend()

    plt.tight_layout()
    plt.savefig('lstm_results.png')
    plt.show()


# ------------------ NEW PART: Optuna Objective Function ------------------

def objective(trial, X_train, y_train, X_val, y_val, X_test, y_test, scaler_y):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        scaler_y: StandardScaler for y values

    Returns:
        float: Validation loss (to be minimized)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define hyperparameters to optimize
    input_size = 3  # Fixed based on data

    # Suggest hyperparameters using Optuna
    hidden_size = trial.suggest_int("hidden_size", 4, 64)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])

    # Learning rate
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Select optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    # Select loss function
    loss_function = trial.suggest_categorical("loss_function", ["MSE", "MAE", "Huber"])

    # Create model
    model = LSTMRegressor(input_size, hidden_size, num_layers, dropout).to(device)

    # Create datasets and dataloaders
    train_dataset = MultivariateTimeSeriesDataset(X_train, y_train)
    val_dataset = MultivariateTimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Set criterion based on suggested loss function
    if loss_function == "MSE":
        criterion = nn.MSELoss()
    elif loss_function == "MAE":
        criterion = nn.L1Loss()
    else:  # Huber
        criterion = nn.SmoothL1Loss()

    # Set optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Early stopping parameters
    patience = 10
    num_epochs = 100

    # Training
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the best model parameters
            best_model_params = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Load the best model for evaluation
    model.load_state_dict(best_model_params)

    # Report the best validation loss
    trial.report(best_val_loss, epoch)

    # Handle pruning based on the intermediate value
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return best_val_loss


# ------------------ PART 7: Main Function with Optuna ------------------

def main():
    """
    Main function to run the entire pipeline with Optuna hyperparameter optimization.
    """
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    print("Loading data...")
    ground_truth = pd.read_excel('df_CP_5_listlike.xlsx')
    X, y = load_and_prepare_data(ground_truth)

    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler_X, scaler_y = preprocess_data(X, y, test_size=0.1, seed=seed)

    # Split training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)

    print(f"Train shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Validation shapes - X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"Test shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Create a study object and optimize the objective function
    print("Starting hyperparameter optimization with Optuna...")
    start_time = time.perf_counter()

    # Create Optuna study
    study = optuna.create_study(direction="minimize")

    # Run the optimization
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, X_test, y_test, scaler_y),
                   n_trials=500, timeout=3600)  # Run 500 trials or for 1 hour

    end_time = time.perf_counter()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")

    # Get the best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # Train the model with the best hyperparameters
    print("Training model with the best hyperparameters...")

    # Extract best hyperparameters
    input_size = 3  # Fixed based on data
    hidden_size = best_params['hidden_size']
    num_layers = best_params['num_layers']
    dropout = best_params['dropout']
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    optimizer_name = best_params['optimizer']
    loss_function = best_params['loss_function']

    # Create datasets and dataloaders for final model
    train_dataset = MultivariateTimeSeriesDataset(X_train, y_train)
    val_dataset = MultivariateTimeSeriesDataset(X_val, y_val)
    test_dataset = MultivariateTimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model with the best hyperparameters
    best_model = LSTMRegressor(input_size, hidden_size, num_layers, dropout)

    # Set criterion based on the best loss function
    if loss_function == "MSE":
        criterion = nn.MSELoss()
    elif loss_function == "MAE":
        criterion = nn.L1Loss()
    else:  # Huber
        criterion = nn.SmoothL1Loss()

    # Set optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(best_model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(best_model.parameters(), lr=learning_rate)
    else:  # SGD
        optimizer = optim.SGD(best_model.parameters(), lr=learning_rate)

    # Train the model
    start = time.perf_counter()
    trained_model, history = train_model(
        model=best_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=100,
        patience=20
    )
    end = time.perf_counter()
    print(f"Training time: {end - start:.2f} seconds")

    # Evaluate the model
    print("Evaluating model...")
    evaluation_results = evaluate_model(trained_model, test_loader, scaler_y)

    # Visualize results
    print("Visualizing results...")
    plot_results(history, evaluation_results)

    # Plot optimization history
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig('optuna_optimization_history.png')

    # Plot parameter importances
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.tight_layout()
    plt.savefig('optuna_param_importances.png')

    # Plot parallel coordinate plot
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.tight_layout()
    plt.savefig('optuna_parallel_coordinate.png')

    # Return trained model and results
    return trained_model, evaluation_results, study


if __name__ == "__main__":
    main()