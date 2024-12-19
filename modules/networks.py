import torch
import torch.nn as nn

class network_qpidgram(nn.Module):
    """A neural network model that takes combined tensor input and processes it using n-gram and word-based pathways."""
    
    def __init__(self, input_dim: int, len_output: int, dropout_rate: float = 0.5):
        """Initializes the model with embedding, convolution, pooling, and fully connected layers.
        
        Args:
            input_dim (int): The size of the combined vocabulary.
            len_output (int): The number of output classes.
            dropout_rate (float): The dropout rate.
        """
        super(network_qpidgram, self).__init__()
        
        # Embedding layer for the combined tensor input
        self.embedding = nn.Embedding(num_embeddings=input_dim, embedding_dim=128)
        
        # Convolutional pathway for the combined input
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Global max pooling layer
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers after concatenating n-gram and word pathways
        self.fc_block = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, len_output)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass on the input tensor X.
        
        Args:
            X (torch.Tensor): The input tensor of shape (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: The model output tensor.
        """
        # Embedding layer
        x = self.embedding(X)
        x = x.transpose(1, 2)  # Transpose to (batch_size, channels, sequence_length)
        
        # Convolutional pathway
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Global max pooling
        x = self.global_max_pooling(x).squeeze(-1)  # Shape: (batch_size, 1024)
        
        # Fully connected layers
        output = self.fc_block(x)  # Shape: (batch_size, len_output)
        
        return output