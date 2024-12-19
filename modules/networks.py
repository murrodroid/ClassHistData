import torch
import torch.nn as nn

class network_qpidgram(nn.Module):
    def __init__(self, vocab_size, ngram_vocab_size, word_vocab_size, len_output):
        super(network_qpidgram, self).__init__()
        
        # Embedding layers for n-gram and word tokenized inputs
        self.ngram_embedding = nn.Embedding(num_embeddings=ngram_vocab_size, embedding_dim=128)
        self.word_embedding = nn.Embedding(num_embeddings=word_vocab_size, embedding_dim=128)
        
        # Convolution blocks for n-gram pathway
        self.ngram_conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.ngram_conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.ngram_conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Convolution blocks for word pathway
        self.word_conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.word_conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.word_conv_block3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Global max pooling layer for both pathways
        self.global_max_pooling = nn.AdaptiveMaxPool1d(1)

        # Fully connected layers after concatenating n-gram and word pathways
        self.fc_block = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, len_output)
        )

    def forward(self, ngram_input, word_input):
        # N-Gram pathway
        x_ngram = self.ngram_embedding(ngram_input)
        x_ngram = x_ngram.transpose(1, 2)
        x_ngram = self.ngram_conv_block1(x_ngram)
        x_ngram = self.ngram_conv_block2(x_ngram)
        x_ngram = self.ngram_conv_block3(x_ngram)
        x_ngram = self.global_max_pooling(x_ngram).squeeze(-1)
        
        # Word pathway
        x_word = self.word_embedding(word_input)
        x_word = x_word.transpose(1, 2)
        x_word = self.word_conv_block1(x_word)
        x_word = self.word_conv_block2(x_word)
        x_word = self.word_conv_block3(x_word)
        x_word = self.global_max_pooling(x_word).squeeze(-1)
        
        # Concatenate n-gram and word pathways
        x = torch.cat((x_ngram, x_word), dim=1)

        # Fully connected layers
        output = self.fc_block(x)

        return output

