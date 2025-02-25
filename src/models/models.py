import torch
import torch.nn as nn 
import math 
import torch.nn.functional as F


class EnhancedAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_channels=2, sfreq=100, lstm_hidden_size=64, 
                 lstm_layers=2, n_attention_heads=4, dropout=0.25):
        super().__init__()
        
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        
        self.cnn_block = nn.Sequential(
            nn.Conv2d(1, 32, (n_channels, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),
            
            nn.Conv2d(64, 128, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2))
        )
        
        # Calculate the output size after CNN block
        # After two MaxPool2d operations, the temporal dimension is reduced by factor of 4
        self.cnn_temporal_size = input_size // 4
        
        # LSTM block - input size is now 128 (from CNN output channels)
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Multi-head attention block
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden_size * 2,  # *2 because of bidirectional LSTM
            num_heads=n_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Projection head remains the same
        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
    
    def get_embedding(self, x):
        """
        Forward pass through the network up to the embedding.
        Args:
            x (torch.Tensor): Input data of shape (batch, channels, time)
        """
        # Add channel dimension for 2D convolution: (batch, 1, channels, time)
        x = x.unsqueeze(1)
        
        x = self.cnn_block(x.float())
        
        # Reshape for LSTM: (batch, time, channels)
        # Remove the redundant spatial dimension and permute
        x = x.squeeze(2)  # Remove the spatial dimension that was reduced to 1
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        
        lstm_out, _ = self.lstm(x)
        
        attention_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        embedding = attention_out.mean(dim=1)
        
        return embedding
    
    def forward(self, x):
        """
        Forward pass with projection head.
        Args:
            x (torch.Tensor): Input data of shape (batch, channels, time)
        Returns:
            torch.Tensor: Projected embeddings of shape (batch, hidden_size)
        """
        embedding = self.get_embedding(x)
        return self.projection(embedding)