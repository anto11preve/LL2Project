import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_svmlight_file
from sympy.physics.units import minutes
#from torch.backends.cudnn import benchmark
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut
import time
#from random import randint
import optuna
#from optuna.trial import TrialState
import pickle #serialization of objects for disk saving



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


class MultivariateTimeSeriesDataset(Dataset):
    """
    Custom Dataset class for handling multivariate time series data.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMRegressor(nn.Module):
    """
    LSTM model for regression on multivariate time series data.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMRegressor, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        output, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1, :, :]
        last_hidden = self.dropout(last_hidden)
        out = self.fc(last_hidden)
        return out.squeeze(1)


def train_single_fold(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10):
    """
    Train model for a single fold with early stopping.

    Returns:
        tuple: (best_val_loss, trained_model_state)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    best_val_loss = float('inf')
    best_model_state = None
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
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= patience:
                break

    return best_val_loss, best_model_state


def loocv_evaluation(X, y, hyperparams, verbose=False):
    """
    Perform Leave-One-Out Cross-Validation.

    Args:
        X: Input features (n_samples, seq_len, n_features)
        y: Target values (n_samples,)
        hyperparams: Dictionary with model hyperparameters
        verbose: Whether to print progress

    Returns:
        tuple: (mean_val_loss, std_val_loss, predictions, actuals)
    """

    torch.manual_seed(20)
    np.random.seed(20)
    '''
    if the seed is not set, the metrics results will be different from "training" to the test even though it has the same hyperparameters.
    this is because the model weights are initialized randomly and the training process is stochastic, if the seed is not set, the state
    of the random number generator will be different each time, leading to different starting weights.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loo = LeaveOneOut()
    fold_losses = []
    all_predictions = []
    all_actuals = []

    # Extract hyperparameters
    input_size = 3
    hidden_size = hyperparams['hidden_size']
    num_layers = hyperparams['num_layers']
    dropout = hyperparams['dropout']
    batch_size = hyperparams['batch_size']
    learning_rate = hyperparams['learning_rate']
    optimizer_name = hyperparams['optimizer']
    #loss_function = hyperparams['loss_function']
    #loss_function = hyperparams.get('loss_function', 'Huber')
    loss_function = hyperparams.get('loss_function', 'MAE')

    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(X)):
        if verbose and fold_idx % 5 == 0:
            print(f"Processing fold {fold_idx + 1}/{len(X)}")

        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Standardize features for this fold
        num_samples, seq_length, num_features = X_train_fold.shape
        X_train_reshaped = X_train_fold.reshape(-1, num_features)

        scaler_X = StandardScaler()
        X_train_scaled_reshaped = scaler_X.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled_reshaped.reshape(num_samples, seq_length, num_features)

        # Scale validation data with same scaler
        X_val_reshaped = X_val_fold.reshape(-1, num_features)
        X_val_scaled_reshaped = scaler_X.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled_reshaped.reshape(1, seq_length, num_features)  # Only 1 sample in validation

        # Standardize targets for this fold
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train_fold.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val_fold.reshape(-1, 1)).flatten()

        # Create datasets and dataloaders
        train_dataset = MultivariateTimeSeriesDataset(X_train_scaled, y_train_scaled)
        val_dataset = MultivariateTimeSeriesDataset(X_val_scaled, y_val_scaled)

        train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)

        # Create model
        model = LSTMRegressor(input_size, hidden_size, num_layers, dropout)

        # Set criterion
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

        # Train the fold
        val_loss, best_model_state = train_single_fold(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=50, patience=8
        )

        # Load best model and make prediction
        model.load_state_dict(best_model_state)
        model.eval()
        model = model.to(device)

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)

                # Inverse transform to original scale
                pred_original = scaler_y.inverse_transform(outputs.cpu().numpy().reshape(-1, 1)).flatten()[0]
                actual_original = scaler_y.inverse_transform(targets.cpu().numpy().reshape(-1, 1)).flatten()[0]

                all_predictions.append(pred_original)
                all_actuals.append(actual_original)

        fold_losses.append(val_loss)

    # Calculate statistics
    mean_val_loss = np.mean(fold_losses)
    std_val_loss = np.std(fold_losses)

    return mean_val_loss, std_val_loss, np.array(all_predictions), np.array(all_actuals)


def calculate_loocv_benchmark(y):
    """
    Calculate benchmark metrics using LOOCV methodology.
    For each fold, predict 0 and calculate the error.

    Args:
        y: Target values

    Returns:
        dict: Benchmark metrics
    """
    loo = LeaveOneOut()
    fold_errors = []
    all_predictions = []
    all_actuals = []

    for train_idx, val_idx in loo.split(y):
        y_val = y[val_idx][0]  # Single validation sample
        prediction = 0.0  # Always predict 0

        all_predictions.append(prediction)
        all_actuals.append(y_val)

        error = (prediction - y_val) ** 2
        fold_errors.append(error)

    # Calculate metrics
    predictions = np.array(all_predictions)
    actuals = np.array(all_actuals)

    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    return {
        "benchmark_mse": mse,
        "benchmark_rmse": rmse,
        "benchmark_mae": mae,
        "benchmark_predictions": predictions,
        "benchmark_actuals": actuals
    }


def objective_loocv(trial, X, y):
    """
    Objective function for Optuna using LOOCV.

    Args:
        trial: Optuna trial object
        X: Input features
        y: Target values

    Returns:
        float: Mean validation loss across all folds
    """
    # Suggest hyperparameters
    hyperparams = {
        'hidden_size': trial.suggest_int("hidden_size", 4, 64),
        'num_layers': trial.suggest_int("num_layers", 1, 3),
        'dropout': trial.suggest_float("dropout", 0.0, 0.5),
        'batch_size': trial.suggest_categorical("batch_size", [2, 4, 8]),
        'learning_rate': trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
        #'loss_function': trial.suggest_categorical("loss_function", ["MSE", "MAE", "Huber"])
        #'loss_function': "Huber"  # Fixed loss
        'loss_function': "MAE"  # Fixed loss
    }

    # Perform LOOCV
    mean_val_loss, std_val_loss, predictions, actuals = loocv_evaluation(X, y, hyperparams, verbose=False)

    # Report intermediate value for pruning (use mean loss)
    trial.report(mean_val_loss, 0)

    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    model_mse = np.mean((predictions - actuals) ** 2)
    model_rmse = np.sqrt(model_mse)
    model_mae = np.mean(np.abs(predictions - actuals))



    #return mean_val_loss
    return model_mae


def train_final_model_loocv(X, y, best_hyperparams):
    """
    Train the final model using the best hyperparameters and evaluate with LOOCV.

    Args:
        X: Input features
        y: Target values
        best_hyperparams: Best hyperparameters from optimization

    Returns:
        dict: Final evaluation results
    """
    print("Training final model with LOOCV evaluation...")

    # Perform LOOCV with best hyperparameters
    mean_val_loss, std_val_loss, predictions, actuals = loocv_evaluation(
        X, y, best_hyperparams, verbose=True
    )

    # Calculate model metrics on original scale
    model_mse = np.mean((predictions - actuals) ** 2)
    model_rmse = np.sqrt(model_mse)
    model_mae = np.mean(np.abs(predictions - actuals))

    # Calculate benchmark with LOOCV methodology
    benchmark_results = calculate_loocv_benchmark(y)

    print("\n" + "=" * 50)
    print("FINAL RESULTS WITH LOOCV")
    print("=" * 50)
    print(f"Model Performance (LOOCV):")
    print(f"  Mean CV Loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
    print(f"  MSE: {model_mse:.4f}")
    print(f"  RMSE: {model_rmse:.4f}")
    print(f"  MAE: {model_mae:.4f}")

    print(f"\nBenchmark Performance (Always predict 0):")
    print(f"  MSE: {benchmark_results['benchmark_mse']:.4f}")
    print(f"  RMSE: {benchmark_results['benchmark_rmse']:.4f}")
    print(f"  MAE: {benchmark_results['benchmark_mae']:.4f}")

    print(f"\nImprovement over benchmark:")
    print(
        f"  MSE improvement: {((benchmark_results['benchmark_mse'] - model_mse) / benchmark_results['benchmark_mse'] * 100):.1f}%"
    )
    print(
        f"  RMSE improvement: {((benchmark_results['benchmark_rmse'] - model_rmse) / benchmark_results['benchmark_rmse'] * 100):.1f}%")
    print(
        f"  MAE improvement: {((benchmark_results['benchmark_mae'] - model_mae) / benchmark_results['benchmark_mae'] * 100):.1f}%")

    return {
        "mean_cv_loss": mean_val_loss,
        "std_cv_loss": std_val_loss,
        "model_mse": model_mse,
        "model_rmse": model_rmse,
        "model_mae": model_mae,
        "predictions": predictions,
        "actuals": actuals,
        **benchmark_results
    }


def plot_loocv_results_old(evaluation_results, study=None):
    """
    Plot LOOCV results and optionally Optuna optimization history.
    """
    predictions = evaluation_results["predictions"]
    actuals = evaluation_results["actuals"]
    benchmark_predictions = evaluation_results["benchmark_predictions"]  # Always 0

    # Create figure
    if study is not None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Predictions vs Actual
    ax1.scatter(actuals, predictions, alpha=0.7, s=60, label="Model predictions", color='blue')
    ax1.scatter(actuals, benchmark_predictions, alpha=0.7, s=60, label="Benchmark (0)", color='red', marker='x')

    # Line of equality
    min_val, max_val = min(min(actuals), min(predictions)), max(max(actuals), max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label="Perfect prediction")

    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title(
        f"LOOCV Results\nModel RMSE: {evaluation_results['model_rmse']:.3f} | Benchmark RMSE: {evaluation_results['benchmark_rmse']:.3f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals
    residuals = predictions - actuals
    ax2.scatter(actuals, residuals, alpha=0.7, s=60, color='blue')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8)
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Residuals (Predicted - Actual)")
    ax2.set_title("Residual Plot")
    ax2.grid(True, alpha=0.3)

    if study is not None:
        # Plot 3: Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax3)
        ax3.set_title("Optuna Optimization History")

        # Plot 4: Parameter importances
        optuna.visualization.matplotlib.plot_param_importances(study, ax=ax4)
        ax4.set_title("Parameter Importances")

    plt.tight_layout()
    plt.savefig('loocv_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_loocv_results(evaluation_results, study=None):
    """
    Plot LOOCV results and optionally Optuna optimization history.
    """
    predictions = evaluation_results["predictions"]
    actuals = evaluation_results["actuals"]
    benchmark_predictions = evaluation_results["predictions"] * 0  # Always 0

    # Create main figure for LOOCV results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Predictions vs Actual
    ax1.scatter(actuals, predictions, alpha=0.7, s=60, label="Model predictions", color='blue')
    ax1.scatter(actuals, benchmark_predictions, alpha=0.7, s=60, label="Benchmark (0)", color='red', marker='x')

    # Line of equality
    min_val, max_val = min(min(actuals), min(predictions)), max(max(actuals), max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label="Perfect prediction")

    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title(
        f"LOOCV Results\nModel MAE: {evaluation_results['model_mae']:.3f} | Benchmark MAE: {evaluation_results['benchmark_mae']:.3f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals
    residuals = predictions - actuals
    ax2.scatter(actuals, residuals, alpha=0.7, s=60, color='blue')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8)
    ax2.set_xlabel("Actual Values")
    ax2.set_ylabel("Residuals (Predicted - Actual)")
    ax2.set_title("Residual Plot")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('loocv_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create separate Optuna plots if study is provided
    if study is not None:
        print("Creating Optuna visualization plots...")

        # Plot optimization history
        try:
            fig_history = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title("Optimization History")
            plt.savefig('optimization_history.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not create optimization history plot: {e}")

        # Plot parameter importances
        try:
            fig_importance = optuna.visualization.matplotlib.plot_param_importances(study)
            #plt.title("Parameter Importances")     #image title is set in the plot function
            plt.savefig('parameter_importances.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not create parameter importance plot: {e}")

        # Plot parameter relationships (optional)
        try:
            fig_parallel = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            #plt.title("Parameter Relationships")   #image title is set in the plot function
            plt.savefig('parameter_relationships.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not create parallel coordinate plot: {e}")

        # Print best trial information
        print(f"\nBest trial:")
        print(f"  Value: {study.best_value:.4f}")
        print(f"  Params: {study.best_params}")
        print(f"  Number of trials: {len(study.trials)}")
        print(
            f"  Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(
            f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

def main():
    """
    Main function with LOOCV implementation.
    """
    # Set random seed for reproducibility
    seed = 20
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_timeout = 7200  # Timeout for training in seconds

    # Load data
    print("Loading data...")
    ground_truth = pd.read_excel('df_CP_5_listlike.xlsx')
    X, y = load_and_prepare_data(ground_truth)

    print(f"Data shapes - X: {X.shape}, y: {y.shape}")
    print(f"Using LOOCV: {len(y)} folds")

    # Create Optuna study
    print("Starting hyperparameter optimization with Optuna + LOOCV...")
    start_time = time.perf_counter()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))

    #improvements at iteration 0,1 and 21
    # Run optimization with LOOCV
    study.optimize(
        lambda trial: objective_loocv(trial, X, y),
        n_trials=500000,
        timeout=train_timeout
    )

    end_time = time.perf_counter()

    delta_time = end_time - start_time
    delta_seconds = delta_time % 60
    delta_time = delta_time // 60
    delta_minutes= delta_time%60
    delta_hours = delta_time//60

    print(f"Total time for optimization: {delta_hours:.0f} hours, {delta_minutes:.0f} minutes and {delta_seconds:.0f} seconds")

    #print(f"Optimization completed in {(end_time - start_time)/360:.2f} hours")

    # Get best hyperparameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    print(f"Best LOOCV score: {study.best_value:.4f}")

    with open('best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    with open('best_params.pkl', 'rb') as f:
        best_params = pickle.load(f)

    print("Best hyperparameters loaded from file:", best_params)

    # Train final model and evaluate with LOOCV
    evaluation_results = train_final_model_loocv(X, y, best_params)



    # Plot results
    print("Creating visualizations...")
    plot_loocv_results(evaluation_results, study)

    return evaluation_results, study


if __name__ == "__main__":
    results, study = main()