import torch.nn as nn

class LanguageClassifierModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(LanguageClassifierModel, self).__init__()
        self.kwargs = {"vocab_size": vocab_size, "embed_dim": embed_dim, "num_class": num_class}
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        return self.fc(embedded)