from .base import BaseModel
from modelling.data_model import Data

import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, classes: int, emb_dim: int) -> None:
        super(Network, self).__init__()
        self.embedding = nn.Embedding(classes, emb_dim)

        self.net = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, classes)
        )

    def forward(self, x):
        logits = []

        logit1 = self.net(x)
        logits.append(logit1)

        for i in range(3):
            output = torch.argmax(logit1, dim=1)
            emb = self.embedding(output)
            x = x + emb
            logit = self.net(x)
            logits.append(logit)

        return logits


class ChainedModel(BaseModel):
    def __init__(self, model_name: str, data: Data) -> None:
        super(ChainedModel, self).__init__()
        self.model_name = model_name
        self.data = data

        self.X_train = data.get_X_train()
        self.X_test = data.get_X_test()
        self.y_train = data.get_type_y_train()
        self.y_test = data.get_type_y_test()

        n_unique = len(torch.unique(self.y_train))
        self.mdl = Network(classes=n_unique, emb_dim=self.X_train.shape[1])

    def train(self, epochs:int = 5):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.mdl.parameters(), lr=0.01)

        for epoch in range(epochs):
            for i in range(0, len(self.embeddings), 32):
                X = self.X_train[i:i+32]
                y = self.y_train[i:i+32]
                
                logits = self.mdl(X)
                loss = 0
                for i, logit in enumerate(logits):
                    loss += criterion(logit, y[:, i])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        all_preds = torch.Tensor()

        for i in range(0, len(X), 32):
            x = X[i:i+32]
            logits = self.mdl(x)
            stacked_logits = torch.stack(logits)
            permuted_logits = stacked_logits.permute(1, 0, 2)
            pred = torch.argmax(permuted_logits, dim=1)
            all_preds = torch.cat((all_preds, pred))

        self.predictions = all_preds
        return all_preds

    def print_results(self):
        y_pred = self.predict(self.X_test).numpy()
        y_true = self.y_test.numpy()
        
        # Shape of y_pred and y_true is (n, 4)
        assert y_pred.shape == y_true.shape
        assert y_true.shape[1] == 4, "Chained model needs all 4 predictions"

        # Accuracy depends on chained predictions
        accuracy = []
        for i in range(len(y_pred)):
            correct = 0
            for j in range(4):
                if y_pred[i][j] == y_true[i][j]:
                    correct += 1
                else:
                    break
            accuracy.append(correct/4)

        self.accuracy = sum(accuracy)/len(accuracy)
        print(f"Accuracy: {self.accuracy}")


    def data_transform(self):
        return super().data_transform()
    
