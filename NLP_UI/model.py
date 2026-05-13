import torch
import torch.nn as nn


class Seq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, pad_id=0):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_id = pad_id

        # ===== Encoder =====
        # No padding_idx — matches training (training used plain nn.Embedding)
        self.encoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.encoder_lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True
        )

        # ===== Decoder =====
        self.decoder_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.decoder_lstm = nn.LSTM(
            embedding_dim + hidden_size * 2,
            hidden_size,
            batch_first=True
        )

        # ===== Bahdanau Attention =====
        self.W1 = nn.Linear(hidden_size * 2, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V  = nn.Linear(hidden_size, 1)

        # ===== Output projection =====
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self):
        # Inference is done manually in app.py
        pass
