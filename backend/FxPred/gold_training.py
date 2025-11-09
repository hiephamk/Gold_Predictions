import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, TensorDataset


def create_data_loaders(sequences, targets, batch_size=32, train_split=0.8, shuffle=True):
    """Create train and validation data loaders for OHLC predictions"""
    sequences_tensor = torch.FloatTensor(sequences)
    # Remove unsqueeze since targets already have shape (n_samples, 4)
    target_tensor = torch.FloatTensor(targets)
    dataset = TensorDataset(sequences_tensor, target_tensor)

    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"✓ Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    return train_loader, val_loader


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model, max_len=10000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ForexTransformer(nn.Module):
    """Transformer model for forex/gold OHLC price prediction"""

    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6,
                 dim_feedforward=1024, dropout=0.1, output_dim=4):
        super(ForexTransformer, self).__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Modified output head for OHLC (4 outputs)
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)  # Output: [Open, High, Low, Close]
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]  # Take last time step
        x = self.output_head(x)
        return x


# def train_model(model, train_loader, val_loader, epochs=300, lr=0.001,
#                 weight_decay=1e-4, progress_callback=None):
#     """Train the model with optional progress callback for OHLC prediction"""
#     device = torch.device('mps' if torch.backends.mps.is_available() else
#                           'cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)

#     criterion = nn.MSELoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=10
#     )

#     warmup_epochs = 10
#     warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
#         optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
#     )

#     train_losses = []
#     val_losses = []
#     val_maes = []
#     val_direction_accuracies = []
#     best_val_loss = float('inf')
#     best_direction_acc = 0
#     early_stop_patience = 30
#     epochs_no_improve = 0

#     print(f"\nTraining on {device}...")
#     print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Val MAE':<15} {'Val Dir Acc':<15} {'LR':<12}")
#     print("-" * 85)

#     for epoch in range(epochs):
#         # Training
#         model.train()
#         train_loss = 0
#         for sequences, targets in train_loader:
#             sequences, targets = sequences.to(device), targets.to(device)

#             optimizer.zero_grad()
#             outputs = model(sequences)
#             loss = criterion(outputs, targets)

#             # OHLC constraint: High should be >= Low
#             if epoch >= warmup_epochs:
#                 high_pred = outputs[:, 1]  # High
#                 low_pred = outputs[:, 2]  # Low
#                 constraint_penalty = torch.mean(torch.relu(low_pred - high_pred))
#                 loss += 0.1 * constraint_penalty

#                 # Smoothness penalty on Close prices
#                 if len(outputs) >= 2:
#                     close_pred = outputs[:, 3]  # Close
#                     smoothness_penalty = torch.mean((close_pred[1:] - close_pred[:-1]) ** 2)
#                     loss += 0.01 * smoothness_penalty

#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()

#             train_loss += loss.item()

#         train_loss /= len(train_loader)
#         train_losses.append(train_loss)

#         # Validation
#         model.eval()
#         val_loss = 0
#         val_mae = 0
#         correct_direction = 0
#         total_direction = 0

#         with torch.no_grad():
#             for sequences, targets in val_loader:
#                 sequences, targets = sequences.to(device), targets.to(device)
#                 outputs = model(sequences)

#                 loss = criterion(outputs, targets)
#                 val_loss += loss.item()

#                 # Calculate MAE for all OHLC
#                 val_mae += torch.mean(torch.abs(outputs - targets)).item()

#                 # Directional accuracy based on Close prices
#                 if len(outputs) >= 2:
#                     pred_close = outputs[:, 3]  # Close prices
#                     true_close = targets[:, 3]
#                     pred_diff = pred_close[1:] - pred_close[:-1]
#                     true_diff = true_close[1:] - true_close[:-1]
#                     correct_direction += torch.sum((pred_diff * true_diff) > 0).item()
#                     total_direction += len(pred_diff)

#         val_loss /= len(val_loader)
#         val_mae /= len(val_loader)
#         val_direction_accuracy = (correct_direction / total_direction) * 100 if total_direction > 0 else 0

#         val_losses.append(val_loss)
#         val_maes.append(val_mae)
#         val_direction_accuracies.append(val_direction_accuracy)

#         current_lr = optimizer.param_groups[0]['lr']

#         # Step schedulers
#         if epoch < warmup_epochs:
#             warmup_scheduler.step()
#         else:
#             scheduler.step(val_loss)

#         # Early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1

#         if val_direction_accuracy > best_direction_acc:
#             best_direction_acc = val_direction_accuracy

#         # Print progress
#         if (epoch + 1) % 10 == 0:
#             msg = f"{epoch + 1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {val_mae:<15.6f} {val_direction_accuracy:<15.2f}% {current_lr:<12.6f}"
#             print(msg)
#             if progress_callback:
#                 progress_callback(epoch + 1, epochs, msg)

#         # Early stopping check
#         if epochs_no_improve >= early_stop_patience:
#             print(f"\n✓ Early stopping at epoch {epoch + 1}")
#             break

#     print("\n" + "=" * 85)
#     print(f"✓ Best validation loss: {best_val_loss:.6f}")
#     print(f"✓ Best directional accuracy: {best_direction_acc:.2f}%")
#     print("=" * 85)

#     return model, train_losses, val_losses, val_maes, val_direction_accuracies



# import torch
# import torch.nn as nn
# import math
# from torch.utils.data import DataLoader, TensorDataset
# # from fetch_dataset_yf import (fetch_gold_price, add_technical_indicators, get_feature_columns, prepare_data)
#
#
# def create_data_loaders(sequences, targets, batch_size=32, train_split=0.8, shuffle=True):
#     """Create train and validation data loaders"""
#     sequences_tensor = torch.FloatTensor(sequences)
#     target_tensor = torch.FloatTensor(targets).unsqueeze(1)
#     dataset = TensorDataset(sequences_tensor, target_tensor)
#
#     train_size = int(len(dataset) * train_split)
#     val_size = len(dataset) - train_size
#     train_dataset, val_dataset = torch.utils.data.random_split(
#         dataset, [train_size, val_size],
#         generator=torch.Generator().manual_seed(42)
#     )
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#
#     print(f"✓ Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
#
#     return train_loader, val_loader
#
# class PositionalEncoding(nn.Module):
#     """Positional encoding for transformer"""
#     def __init__(self, d_model, max_len=10000, dropout=0.1):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() *
#                            (-math.log(10000.0) / d_model))
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         if d_model % 2 == 1:
#             pe[:, 1::2] = torch.cos(position * div_term[:-1])
#         else:
#             pe[:, 1::2] = torch.cos(position * div_term)
#
#         pe = pe.unsqueeze(0)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)
#
# class ForexTransformer(nn.Module):
#     """Transformer model for forex/gold price prediction"""
#     def __init__(self, input_dim, d_model=256, nhead=8, num_layers=6,
#                  dim_feedforward=1024, dropout=0.1):
#         super(ForexTransformer, self).__init__()
#
#         if d_model % nhead != 0:
#             raise ValueError("d_model must be divisible by nhead")
#
#         self.input_projection = nn.Sequential(
#             nn.Linear(input_dim, d_model),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
#
#         self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,
#             activation='gelu',
#             norm_first=True
#         )
#
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#
#         self.output_head = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, 256),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, 128),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(128, 64),
#             nn.GLU(),
#             nn.Linear(32, 1)
#         )
#
#     def forward(self, x):
#         x = self.input_projection(x)
#         x = self.pos_encoder(x)
#         x = self.transformer_encoder(x)
#         x = x[:, -1, :]
#         x = self.output_head(x)
#         return x
#
#
# def train_model(model, train_loader, val_loader, epochs=300, lr=0.001,
#                 weight_decay=1e-4, progress_callback=None):
#     """Train the model with optional progress callback"""
#     device = torch.device('mps' if torch.backends.mps.is_available() else
#                          'cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=10
#     )
#
#     warmup_epochs = 10
#     warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
#         optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
#     )
#
#     train_losses = []
#     val_losses = []
#     val_maes = []
#     val_direction_accuracies = []
#     best_val_loss = float('inf')
#     best_direction_acc = 0
#     early_stop_patience = 30
#     epochs_no_improve = 0
#
#     print(f"\nTraining on {device}...")
#     print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Val MAE':<15} {'Val Dir Acc':<15} {'LR':<12}")
#     print("-" * 85)
#
#     for epoch in range(epochs):
#         # Training
#         model.train()
#         train_loss = 0
#         for sequences, targets in train_loader:
#             sequences, targets = sequences.to(device), targets.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(sequences)
#             loss = criterion(outputs, targets)
#
#             if epoch >= warmup_epochs and len(outputs) >= 2:
#                 outputs_squeezed = outputs.squeeze()
#                 smoothness_penalty = torch.mean((outputs_squeezed[1:] - outputs_squeezed[:-1]) ** 2)
#                 loss += 0.01 * smoothness_penalty
#
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
#
#             train_loss += loss.item()
#
#         train_loss /= len(train_loader)
#         train_losses.append(train_loss)
#
#         # Validation
#         model.eval()
#         val_loss = 0
#         val_mae = 0
#         correct_direction = 0
#         total_direction = 0
#
#         with torch.no_grad():
#             for sequences, targets in val_loader:
#                 sequences, targets = sequences.to(device), targets.to(device)
#                 outputs = model(sequences)
#
#                 loss = criterion(outputs, targets)
#                 val_loss += loss.item()
#                 val_mae += torch.mean(torch.abs(outputs - targets)).item()
#
#                 if len(outputs) >= 2:
#                     pred_vals = outputs.squeeze()
#                     true_vals = targets.squeeze()
#                     pred_diff = pred_vals[1:] - pred_vals[:-1]
#                     true_diff = true_vals[1:] - true_vals[:-1]
#                     correct_direction += torch.sum((pred_diff * true_diff) > 0).item()
#                     total_direction += len(pred_diff)
#
#         val_loss /= len(val_loader)
#         val_mae /= len(val_loader)
#         val_direction_accuracy = (correct_direction / total_direction) * 100 if total_direction > 0 else 0
#
#         val_losses.append(val_loss)
#         val_maes.append(val_mae)
#         val_direction_accuracies.append(val_direction_accuracy)
#
#         current_lr = optimizer.param_groups[0]['lr']
#
#         # Step schedulers
#         if epoch < warmup_epochs:
#             warmup_scheduler.step()
#         else:
#             scheduler.step(val_loss)
#
#         # Early stopping
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#
#         if val_direction_accuracy > best_direction_acc:
#             best_direction_acc = val_direction_accuracy
#
#         # Print progress
#         if (epoch + 1) % 10 == 0:
#             msg = f"{epoch+1:<8} {train_loss:<15.6f} {val_loss:<15.6f} {val_mae:<15.6f} {val_direction_accuracy:<15.2f}% {current_lr:<12.6f}"
#             print(msg)
#             if progress_callback:
#                 progress_callback(epoch + 1, epochs, msg)
#
#         # Early stopping check
#         if epochs_no_improve >= early_stop_patience:
#             print(f"\n✓ Early stopping at epoch {epoch+1}")
#             break
#
#     print("\n" + "=" * 85)
#     print(f"✓ Best validation loss: {best_val_loss:.6f}")
#     print(f"✓ Best directional accuracy: {best_direction_acc:.2f}%")
#     print("=" * 85)
#
#     return model, train_losses, val_losses, val_maes, val_direction_accuracies