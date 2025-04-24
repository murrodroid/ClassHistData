import torch
import torch.nn as nn

class network_qpidgram(nn.Module):
    """A neural network model that takes combined tensor input and processes it using n-gram and word-based pathways."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.5):
        """Initializes the model with embedding, convolution, pooling, and fully connected layers.
        
        Args:
            input_dim (int): The size of the combined vocabulary.
            output_dim (int): The number of output classes.
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
            nn.Linear(1024, output_dim)
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

class individualized_network(nn.Module):
    """
    A neural network model that processes:
      - Cause-of-death text (via CNN),
      - Age (via small feed-forward),
      - Sex (via embedding)
    and concatenates them for final classification.
    """

    def __init__(
        self,
        vocab_size_cause: int,
        num_classes: int,
        dropout_rate: float = 0.5,
        emb_dim_cause: int = 128,
        emb_dim_sex: int = 8,
        emb_dim_age: int = 8
    ):
        """
        Args:
            vocab_size_cause (int): Size of the cause-of-death vocabulary.
            num_classes (int): Number of output classes.
            dropout_rate (float): Dropout rate.
            emb_dim_cause (int): Dimension of the cause-of-death embedding.
            emb_dim_sex (int): Dimension of the sex embedding.
            emb_dim_age (int): Dimension of the age embedding or projection.
        """
        super(individualized_network, self).__init__()

        # ------------------------------------------------------------
        # 1) Embedding + CNN pipeline for cause-of-death
        # ------------------------------------------------------------
        self.embedding_cause = nn.Embedding(
            num_embeddings=vocab_size_cause,
            embedding_dim=emb_dim_cause
        )

        # Convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=emb_dim_cause, out_channels=256, kernel_size=3, padding=1),
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

        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)

        # ------------------------------------------------------------
        # 2) Embeddings / Projections for Age and Sex
        # ------------------------------------------------------------
    
        self.age_fc = nn.Sequential(
            nn.Linear(1, emb_dim_age),
            nn.ReLU()
        )
        # For sex (categorical), using an embedding is common:
        self.sex_embedding = nn.Embedding(
            num_embeddings=8,  # a small upper bound for sex categories
            embedding_dim=emb_dim_sex
        )

        # ------------------------------------------------------------
        # 3) Fully connected layers to combine CNN + Age + Sex
        # ------------------------------------------------------------
        combined_dim = 1024 + emb_dim_age + emb_dim_sex

        self.fc_block = nn.Sequential(
            nn.Linear(combined_dim, 2048),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x_cause: torch.Tensor, x_age: torch.Tensor, x_sex: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_cause: (batch_size, sequence_length) text input for cause-of-death.
            x_age:   (batch_size,) or (batch_size, 1) numeric or integer-coded age.
            x_sex:   (batch_size,) integer-coded sex categories.

        Returns:
            (batch_size, num_classes)
        """
        # --------------------------
        # CNN pipeline for cause-of-death
        # --------------------------
        # Embedding
        x_embed_cause = self.embedding_cause(x_cause)  # (B, seq_len, emb_dim_cause)
        x_embed_cause = x_embed_cause.transpose(1, 2)  # (B, emb_dim_cause, seq_len)

        # Convolutional blocks
        x_embed_cause = self.conv_block1(x_embed_cause)
        x_embed_cause = self.conv_block2(x_embed_cause)
        x_embed_cause = self.conv_block3(x_embed_cause)

        # Global max pooling => (B, 1024)
        x_embed_cause = self.global_max_pooling(x_embed_cause).squeeze(-1)

        # --------------------------
        # Process Age
        # --------------------------
        x_age_emb = self.age_fc(x_age.float())  # shape: (B, emb_dim_age)

        # --------------------------
        # Process Sex
        # --------------------------
        x_sex_emb = self.sex_embedding(x_sex)  # shape: (B, emb_dim_sex)

        # --------------------------
        # Combine
        # --------------------------
        # Concat [CNN cause, age embedding, sex embedding]
        x_concat = torch.cat([x_embed_cause, x_age_emb, x_sex_emb], dim=1)

        # Fully connected block => final output
        output = self.fc_block(x_concat)
        return output
