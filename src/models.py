import torch.nn as nn
from config import NUM_LAYERS, DROPOUT, MODEL_REGISTRY


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def generate_model(model_name: str, *args, **kwargs):
    try:
        model_cls = MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(f"Model '{model_name}' is not recognized.")
    return model_cls(*args, **kwargs)


@register_model("LSTMArabicModel")
class LSTMArabicModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, PAD):
        super(LSTMArabicModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            batch_first=True, bidirectional=True,
                            num_layers=NUM_LAYERS, dropout=DROPOUT)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output
