import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import time
import optuna
import pickle
from typing import Dict, Any, Tuple, List


def load_and_prepare_data(df_CP_5):
    df = df_CP_5.copy()

    def parse_series(s):
        return np.array(ast.literal_eval(s), dtype=np.float32)

    features = ['heart_rate_data', 'respiration_data', 'stress_data']
    for col in features:
        df[col] = df[col].apply(parse_series)

    X = np.stack(df[features].apply(lambda row: np.stack(row, axis=-1), axis=1).values)
    y = df['Target_Unbiased'].astype(np.float32).values
    return X, y


class MultivariateTimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RNNBlock(nn.Module):
    """RNN + BatchNorm + Pooling block"""
    def __init__(self, input_size, hidden_size, rnn_type='LSTM', pooling_type='max'):
        super(RNNBlock, self).__init__()
        
        self.rnn_type = rnn_type
        self.pooling_type = pooling_type
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:  # RNN
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
            
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        if self.rnn_type in ['LSTM']:
            output, (hidden, cell) = self.rnn(x)
        else:
            output, hidden = self.rnn(x)
            
        # Apply batch normalization (reshape for BatchNorm1d)
        batch_size, seq_len, hidden_size = output.shape
        output = output.reshape(batch_size * seq_len, hidden_size)
        output = self.batch_norm(output)
        output = output.reshape(batch_size, seq_len, hidden_size)
        
        # Apply pooling
        if self.pooling_type == 'max':
            pooled_output, _ = torch.max(output, dim=1)  # Max over sequence length
        elif self.pooling_type == 'avg':
            pooled_output = torch.mean(output, dim=1)  # Average over sequence length
        else:  # 'last'
            pooled_output = output[:, -1, :]  # Last timestep
            
        return output, pooled_output


class LSTMBlock(nn.Module):
    """LSTM + Dropout block"""
    def __init__(self, input_size, hidden_size, dropout_rate=0.2):
        super(LSTMBlock, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        output = self.dropout(output)
        return output


class AttentionLayer(nn.Module):
    """Multi-head attention layer"""
    def __init__(self, hidden_size, num_heads=4, dropout_rate=0.1):
        super(AttentionLayer, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
    def forward(self, x):
        attended_output, attention_weights = self.attention(x, x, x)
        return attended_output


class ModularLSTMRegressor(nn.Module):
    """
    Modular architecture: n * (RNN+BatchNorm+Pooling) + m * (LSTM+Dropout) + FC
    Attention layers can be placed at different positions
    """
    def __init__(self, config: Dict[str, Any]):
        super(ModularLSTMRegressor, self).__init__()
        
        # Extract configuration
        self.input_size = config['input_size']
        self.n_rnn_blocks = config['n_rnn_blocks']
        self.m_lstm_blocks = config['m_lstm_blocks']
        self.rnn_hidden_sizes = config['rnn_hidden_sizes']
        self.lstm_hidden_sizes = config['lstm_hidden_sizes']
        self.rnn_types = config['rnn_types']
        self.pooling_types = config['pooling_types']
        self.dropout_rates = config['dropout_rates']
        self.use_attention = config['use_attention']
        self.attention_positions = config['attention_positions']
        self.attention_heads = config['attention_heads']
        self.fc_hidden_size = config['fc_hidden_size']
        self.fc_dropout = config['fc_dropout']
        
        # Build RNN blocks
        self.rnn_blocks = nn.ModuleList()
        current_input_size = self.input_size
        
        for i in range(self.n_rnn_blocks):
            block = RNNBlock(
                input_size=current_input_size,
                hidden_size=self.rnn_hidden_sizes[i],
                rnn_type=self.rnn_types[i],
                pooling_type=self.pooling_types[i]
            )
            self.rnn_blocks.append(block)
            current_input_size = self.rnn_hidden_sizes[i]
        
        # Attention after RNN blocks
        self.attention_after_rnn = None
        if self.use_attention and 'after_rnn' in self.attention_positions:
            self.attention_after_rnn = AttentionLayer(
                current_input_size, 
                self.attention_heads
            )
        
        # Build LSTM blocks
        self.lstm_blocks = nn.ModuleList()
        
        for i in range(self.m_lstm_blocks):
            block = LSTMBlock(
                input_size=current_input_size,
                hidden_size=self.lstm_hidden_sizes[i],
                dropout_rate=self.dropout_rates[i]
            )
            self.lstm_blocks.append(block)
            current_input_size = self.lstm_hidden_sizes[i]
        
        # Attention before FC
        self.attention_before_fc = None
        if self.use_attention and 'before_fc' in self.attention_positions:
            self.attention_before_fc = AttentionLayer(
                current_input_size, 
                self.attention_heads
            )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(current_input_size, self.fc_hidden_size),
            nn.ReLU(),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_hidden_size, self.fc_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.fc_dropout),
            nn.Linear(self.fc_hidden_size // 2, 1)
        )
        
        # Attention at the beginning
        self.attention_at_start = None
        if self.use_attention and 'at_start' in self.attention_positions:
            self.attention_at_start = AttentionLayer(
                self.input_size, 
                self.attention_heads
            )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Attention at start
        if self.attention_at_start is not None:
            x = self.attention_at_start(x)
        
        # Pass through RNN blocks
        pooled_outputs = []
        current_output = x
        
        for i, rnn_block in enumerate(self.rnn_blocks):
            full_output, pooled_output = rnn_block(current_output)
            pooled_outputs.append(pooled_output)
            current_output = full_output
        
        # Attention after RNN blocks
        if self.attention_after_rnn is not None:
            current_output = self.attention_after_rnn(current_output)
        
        # Pass through LSTM blocks
        for lstm_block in self.lstm_blocks:
            current_output = lstm_block(current_output)
        
        # Attention before FC
        if self.attention_before_fc is not None:
            current_output = self.attention_before_fc(current_output)
        
        # Get final representation (last timestep)
        final_hidden = current_output[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc_layers(final_hidden)
        
        return output.squeeze(1)


def create_model_config(trial) -> Dict[str, Any]:
    """Create model configuration from Optuna trial"""
    
    # Basic architecture parameters
    n_rnn_blocks = trial.suggest_int("n_rnn_blocks", 1, 4)
    m_lstm_blocks = trial.suggest_int("m_lstm_blocks", 1, 3)
    
    # RNN block configurations
    rnn_hidden_sizes = []
    rnn_types = []
    pooling_types = []
    
    for i in range(n_rnn_blocks):
        rnn_hidden_sizes.append(trial.suggest_int(f"rnn_hidden_size_{i}", 16, 128, step=16))
        rnn_types.append(trial.suggest_categorical(f"rnn_type_{i}", ["LSTM", "GRU", "RNN"]))
        pooling_types.append(trial.suggest_categorical(f"pooling_type_{i}", ["max", "avg", "last"]))
    
    # LSTM block configurations
    lstm_hidden_sizes = []
    dropout_rates = []
    
    for i in range(m_lstm_blocks):
        lstm_hidden_sizes.append(trial.suggest_int(f"lstm_hidden_size_{i}", 16, 128, step=16))
        dropout_rates.append(trial.suggest_float(f"lstm_dropout_{i}", 0.0, 0.7))
    
    # Attention configuration
    use_attention = trial.suggest_categorical("use_attention", [True, False])
    attention_positions = []
    attention_heads = 4
    
    if use_attention:
        # Suggest which positions to use attention
        if trial.suggest_categorical("attention_at_start", [True, False]):
            attention_positions.append("at_start")
        if trial.suggest_categorical("attention_after_rnn", [True, False]):
            attention_positions.append("after_rnn")
        if trial.suggest_categorical("attention_before_fc", [True, False]):
            attention_positions.append("before_fc")
        
        attention_heads = trial.suggest_categorical("attention_heads", [2, 4, 8])
        
        # Ensure at least one attention position if attention is enabled
        if not attention_positions:
            attention_positions = ["before_fc"]  # Default position
    
    # Fully connected layer configuration
    fc_hidden_size = trial.suggest_int("fc_hidden_size", 32, 256, step=32)
    fc_dropout = trial.suggest_float("fc_dropout", 0.0, 0.5)
    
    return {
        'input_size': 3,  # Fixed for your data
        'n_rnn_blocks': n_rnn_blocks,
        'm_lstm_blocks': m_lstm_blocks,
        'rnn_hidden_sizes': rnn_hidden_sizes,
        'lstm_hidden_sizes': lstm_hidden_sizes,
        'rnn_types': rnn_types,
        'pooling_types': pooling_types,
        'dropout_rates': dropout_rates,
        'use_attention': use_attention,
        'attention_positions': attention_positions,
        'attention_heads': attention_heads,
        'fc_hidden_size': fc_hidden_size,
        'fc_dropout': fc_dropout
    }


def create_training_config(trial) -> Dict[str, Any]:
    """Create training configuration from Optuna trial"""
    
    return {
        'batch_size': trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16]),
        'learning_rate': trial.suggest_float("learning_rate", 1e-6, 1e-1, log=True),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "AdamW"]),
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        'scheduler_type': trial.suggest_categorical("scheduler", ["none", "step", "cosine", "plateau"]),
        'patience': trial.suggest_int("patience", 5, 20),
        'num_epochs': trial.suggest_int("num_epochs", 30, 100),
        'loss_function': trial.suggest_categorical("loss_function", ["MSE", "MAE", "Huber"]),
    }


def create_optimizer_and_scheduler(model, config):
    """Create optimizer and scheduler based on configuration"""
    
    if config['optimizer'] == "Adam":
        optimizer = optim.Adam(model.parameters(), 
                             lr=config['learning_rate'], 
                             weight_decay=config['weight_decay'])
    elif config['optimizer'] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), 
                              lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'])
    elif config['optimizer'] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), 
                                lr=config['learning_rate'], 
                                weight_decay=config['weight_decay'])
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), 
                            lr=config['learning_rate'], 
                            weight_decay=config['weight_decay'])
    
    scheduler = None
    if config['scheduler_type'] == "step":
        step_size = config.get('step_size', 20)
        gamma = config.get('gamma', 0.5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif config['scheduler_type'] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])
    elif config['scheduler_type'] == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    return optimizer, scheduler


def train_single_fold_advanced(model, train_loader, val_loader, config):
    """Advanced training function with schedulers and improved early stopping"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create loss function
    if config['loss_function'] == "MSE":
        criterion = nn.MSELoss()
    elif config['loss_function'] == "MAE":
        criterion = nn.L1Loss()
    else:  # Huber
        criterion = nn.SmoothL1Loss()
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        # Update scheduler
        if scheduler is not None:
            if config['scheduler_type'] == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
            if counter >= config['patience']:
                break
    
    return best_val_loss, best_model_state


def loocv_evaluation_advanced(X, y, model_config, training_config, verbose=False):
    """Advanced LOOCV evaluation with modular architecture"""
    
    torch.manual_seed(20)
    np.random.seed(20)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loo = LeaveOneOut()
    fold_losses = []
    all_predictions = []
    all_actuals = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(X)):
        if verbose and fold_idx % 10 == 0:
            print(f"Processing fold {fold_idx + 1}/{len(X)}")
        
        # Split data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Standardize features
        num_samples, seq_length, num_features = X_train_fold.shape
        X_train_reshaped = X_train_fold.reshape(-1, num_features)
        
        scaler_X = StandardScaler()
        X_train_scaled_reshaped = scaler_X.fit_transform(X_train_reshaped)
        X_train_scaled = X_train_scaled_reshaped.reshape(num_samples, seq_length, num_features)
        
        X_val_reshaped = X_val_fold.reshape(-1, num_features)
        X_val_scaled_reshaped = scaler_X.transform(X_val_reshaped)
        X_val_scaled = X_val_scaled_reshaped.reshape(1, seq_length, num_features)
        
        # Standardize targets
        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_train_fold.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_y.transform(y_val_fold.reshape(-1, 1)).flatten()
        
        # Create datasets and dataloaders
        train_dataset = MultivariateTimeSeriesDataset(X_train_scaled, y_train_scaled)
        val_dataset = MultivariateTimeSeriesDataset(X_val_scaled, y_val_scaled)
        
        train_loader = DataLoader(train_dataset, 
                                batch_size=min(training_config['batch_size'], len(train_dataset)), 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1)
        
        # Create model
        model = ModularLSTMRegressor(model_config)
        
        # Train the fold
        val_loss, best_model_state = train_single_fold_advanced(
            model, train_loader, val_loader, training_config
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


class EarlyStoppingCallback:
    """Early stopping callback for Optuna optimization"""
    def __init__(self, patience=100, min_improvement=1e-5):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_value = float('inf')
        self.counter = 0
    
    def __call__(self, study, trial):
        if trial.value < self.best_value - self.min_improvement:
            self.best_value = trial.value
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            print(f"Early stopping triggered after {self.patience} trials without improvement")
            study.stop()


def objective_advanced(trial, X, y):
    """Advanced objective function with modular architecture"""
    
    # Create configurations
    model_config = create_model_config(trial)
    training_config = create_training_config(trial)
    
    # Add additional scheduler parameters if needed
    if training_config['scheduler_type'] == "step":
        training_config['step_size'] = trial.suggest_int("step_size", 10, 30)
        training_config['gamma'] = trial.suggest_float("gamma", 0.1, 0.9)
    
    try:
        # Perform LOOCV
        mean_val_loss, std_val_loss, predictions, actuals = loocv_evaluation_advanced(
            X, y, model_config, training_config, verbose=False
        )
        
        # Calculate MAE on original scale
        model_mae = np.mean(np.abs(predictions - actuals))
        
        # Report intermediate value for pruning
        trial.report(model_mae, 0)
        
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return model_mae
        
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        return float('inf')  # Return high value for failed trials


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


def train_final_model_advanced(X, y, best_params):
    """
    Train the final model using the best hyperparameters and evaluate with LOOCV.

    Args:
        X: Input features
        y: Target values
        best_params: Best hyperparameters from optimization

    Returns:
        dict: Final evaluation results
    """
    print("Training final model with LOOCV evaluation...")
    
    # Reconstruct model and training configs from best_params
    model_config = {}
    training_config = {}
    
    # Extract model configuration
    model_config['input_size'] = 3
    model_config['n_rnn_blocks'] = best_params['n_rnn_blocks']
    model_config['m_lstm_blocks'] = best_params['m_lstm_blocks']
    
    # Extract RNN block configurations
    model_config['rnn_hidden_sizes'] = []
    model_config['rnn_types'] = []
    model_config['pooling_types'] = []
    
    for i in range(model_config['n_rnn_blocks']):
        model_config['rnn_hidden_sizes'].append(best_params[f'rnn_hidden_size_{i}'])
        model_config['rnn_types'].append(best_params[f'rnn_type_{i}'])
        model_config['pooling_types'].append(best_params[f'pooling_type_{i}'])
    
    # Extract LSTM block configurations
    model_config['lstm_hidden_sizes'] = []
    model_config['dropout_rates'] = []
    
    for i in range(model_config['m_lstm_blocks']):
        model_config['lstm_hidden_sizes'].append(best_params[f'lstm_hidden_size_{i}'])
        model_config['dropout_rates'].append(best_params[f'lstm_dropout_{i}'])
    
    # Extract attention configuration
    model_config['use_attention'] = best_params['use_attention']
    model_config['attention_positions'] = []
    
    if model_config['use_attention']:
        if best_params.get('attention_at_start', False):
            model_config['attention_positions'].append('at_start')
        if best_params.get('attention_after_rnn', False):
            model_config['attention_positions'].append('after_rnn')
        if best_params.get('attention_before_fc', False):
            model_config['attention_positions'].append('before_fc')
        
        model_config['attention_heads'] = best_params.get('attention_heads', 4)
    else:
        model_config['attention_heads'] = 4
    
    # FC configuration
    model_config['fc_hidden_size'] = best_params['fc_hidden_size']
    model_config['fc_dropout'] = best_params['fc_dropout']
    
    # Extract training configuration
    training_config['batch_size'] = best_params['batch_size']
    training_config['learning_rate'] = best_params['learning_rate']
    training_config['optimizer'] = best_params['optimizer']
    training_config['weight_decay'] = best_params['weight_decay']
    training_config['scheduler_type'] = best_params['scheduler']
    training_config['patience'] = best_params['patience']
    training_config['num_epochs'] = best_params['num_epochs']
    training_config['loss_function'] = best_params['loss_function']
    
    if training_config['scheduler_type'] == 'step':
        training_config['step_size'] = best_params.get('step_size', 20)
        training_config['gamma'] = best_params.get('gamma', 0.5)

    # Perform LOOCV with best hyperparameters
    mean_val_loss, std_val_loss, predictions, actuals = loocv_evaluation_advanced(
        X, y, model_config, training_config, verbose=True
    )

    # Calculate model metrics on original scale
    model_mse = np.mean((predictions - actuals) ** 2)
    model_rmse = np.sqrt(model_mse)
    model_mae = np.mean(np.abs(predictions - actuals))

    # Calculate benchmark with LOOCV methodology
    benchmark_results = calculate_loocv_benchmark(y)

    print("\n" + "=" * 50)
    print("FINAL RESULTS WITH ADVANCED LOOCV")
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


def plot_advanced_results(evaluation_results, study=None):
    """
    Plot advanced LOOCV results and optionally Optuna optimization history.
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
        f"Advanced LOOCV Results\nModel MAE: {evaluation_results['model_mae']:.3f} | Benchmark MAE: {evaluation_results['benchmark_mae']:.3f}")
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
    plt.savefig('advanced_loocv_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create separate Optuna plots if study is provided
    if study is not None:
        print("Creating advanced Optuna visualization plots...")

        # Plot optimization history
        try:
            fig_history = optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.title("Advanced Optimization History")
            plt.savefig('advanced_optimization_history.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not create optimization history plot: {e}")

        # Plot parameter importances
        try:
            fig_importance = optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig('advanced_parameter_importances.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not create parameter importance plot: {e}")

        # Plot parameter relationships
        try:
            fig_parallel = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.savefig('advanced_parameter_relationships.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Could not create parallel coordinate plot: {e}")

        # Plot slice plot for key parameters
        try:
            # Focus on most important parameters
            key_params = ['learning_rate', 'n_rnn_blocks', 'm_lstm_blocks', 'fc_hidden_size']
            existing_params = [p for p in key_params if p in study.best_params]
            
            if existing_params:
                fig_slice = optuna.visualization.matplotlib.plot_slice(study, params=existing_params)
                plt.savefig('advanced_parameter_slices.png', dpi=300, bbox_inches='tight')
                plt.show()
        except Exception as e:
            print(f"Could not create slice plot: {e}")

        # Print comprehensive trial information
        print(f"\n" + "=" * 60)
        print("OPTUNA OPTIMIZATION SUMMARY")
        print("=" * 60)
        print(f"Best trial:")
        print(f"  Value (MAE): {study.best_value:.4f}")
        print(f"  Number of trials: {len(study.trials)}")
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
        
        print(f"  Completed trials: {len(completed_trials)}")
        print(f"  Pruned trials: {len(pruned_trials)}")
        print(f"  Failed trials: {len(failed_trials)}")
        
        print(f"\nBest Architecture:")
        
        # Print model architecture details
        best_params = study.best_params
        print(f"  RNN Blocks: {best_params['n_rnn_blocks']}")
        for i in range(best_params['n_rnn_blocks']):
            print(f"    Block {i+1}: {best_params[f'rnn_type_{i}']} - {best_params[f'rnn_hidden_size_{i}']} units - {best_params[f'pooling_type_{i}']} pooling")
        
        print(f"  LSTM Blocks: {best_params['m_lstm_blocks']}")
        for i in range(best_params['m_lstm_blocks']):
            print(f"    Block {i+1}: {best_params[f'lstm_hidden_size_{i}']} units - {best_params[f'lstm_dropout_{i}']:.3f} dropout")
        
        if best_params['use_attention']:
            print(f"  Attention: Enabled")
            print(f"    Heads: {best_params.get('attention_heads', 4)}")
            positions = []
            if best_params.get('attention_at_start', False):
                positions.append('start')
            if best_params.get('attention_after_rnn', False):
                positions.append('after_rnn')
            if best_params.get('attention_before_fc', False):
                positions.append('before_fc')
            print(f"    Positions: {', '.join(positions)}")
        else:
            print(f"  Attention: Disabled")
        
        print(f"  FC Layer: {best_params['fc_hidden_size']} units - {best_params['fc_dropout']:.3f} dropout")
        
        print(f"\nBest Training Config:")
        print(f"  Optimizer: {best_params['optimizer']}")
        print(f"  Learning Rate: {best_params['learning_rate']:.6f}")
        print(f"  Weight Decay: {best_params['weight_decay']:.6f}")
        print(f"  Batch Size: {best_params['batch_size']}")
        print(f"  Scheduler: {best_params['scheduler']}")
        print(f"  Loss Function: {best_params['loss_function']}")
        print(f"  Patience: {best_params['patience']}")
        print(f"  Max Epochs: {best_params['num_epochs']}")


def main_advanced():
    """Main function with advanced architecture"""
    
    # Set random seed for reproducibility
    seed = 20
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_timeout = 200  # 2 hours
    
    # Load data
    print("Loading data...")
    ground_truth = pd.read_excel('df_CP_5_listlike.xlsx')
    X, y = load_and_prepare_data(ground_truth)
    
    print(f"Data shapes - X: {X.shape}, y: {y.shape}")
    print(f"Using LOOCV: {len(y)} folds")
    
    # Create Optuna study with advanced sampler
    print("Starting hyperparameter optimization with advanced architecture...")
    start_time = time.perf_counter()
    
    # Use CmaEs sampler for better continuous optimization
    study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.CmaEsSampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    # Add early stopping callback
    early_stopping = EarlyStoppingCallback(patience=150, min_improvement=1e-4)
    
    # Run optimization
    study.optimize(
        lambda trial: objective_advanced(trial, X, y),
        n_trials=1000,
        timeout=train_timeout,
        callbacks=[early_stopping]
    )
    
    end_time = time.perf_counter()
    delta_time = end_time - start_time
    delta_seconds = delta_time % 60
    delta_time = delta_time // 60
    delta_minutes = delta_time % 60
    delta_hours = delta_time // 60
    
    print(f"Total time for optimization: {delta_hours:.0f} hours, {delta_minutes:.0f} minutes and {delta_seconds:.0f} seconds")
    
    # Save and load best parameters
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    print(f"Best LOOCV MAE: {study.best_value:.4f}")
    
    # Save results
    with open('best_params_advanced.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    
    with open('study_advanced.pkl', 'wb') as f:
        pickle.dump(study, f)
    
    # Train final model and evaluate with LOOCV
    evaluation_results = train_final_model_advanced(X, y, best_params)
    
    # Plot results
    print("Creating advanced visualizations...")
    plot_advanced_results(evaluation_results, study)
    
    return evaluation_results, study


if __name__ == "__main__":
    results, study = main_advanced()