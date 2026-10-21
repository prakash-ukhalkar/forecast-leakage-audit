"""Small, CPU-friendly PyTorch recurrent models (LSTM, GRU) for sequence forecasting.

`torch` is imported lazily inside method bodies rather than at module level, so this
module can still be imported in environments where `torch` is not installed --
consistent with `utils/seed.py`'s graceful degradation. Both forecasters expect
input of shape `(n_samples, seq_len, n_features)`, where `seq_len` is the number of
lag features ordered by increasing lag (see `evaluation.runner`, which builds this
sequence from each dataset's engineered lag columns) and `n_features` is 1.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class _BaseRecurrentForecaster:
    """Shared fit/predict logic for the small LSTM/GRU forecasters."""

    _torch_cell_cls_name: str = ""

    def __init__(self, hidden_size: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self._model: Any = None

    def _build(self, n_features: int) -> Any:
        import torch.nn as nn

        rnn_cls = getattr(nn, self._torch_cell_cls_name)
        hidden_size, num_layers, dropout = self.hidden_size, self.num_layers, self.dropout

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = rnn_cls(
                    input_size=n_features,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0.0,
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return _Net()

    def fit(self, X: Any, y: Any, epochs: int = 30) -> _BaseRecurrentForecaster:
        """Fit the recurrent model on training sequences via full-batch Adam/MSE.

        Args:
            X: Training input sequences, shape (n_samples, seq_len, n_features).
            y: Training targets, shape (n_samples,).
            epochs: Number of full-batch training epochs.

        Returns:
            self.
        """
        import torch

        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32)

        self._model = self._build(n_features=X_t.shape[-1])
        optimizer = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        self._model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = loss_fn(self._model(X_t), y_t)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict on new input sequences.

        Args:
            X: Input sequences, shape (n_samples, seq_len, n_features).

        Returns:
            1-D array of predicted values, length n_samples.
        """
        import torch

        if self._model is None:
            raise RuntimeError("Call .fit() before .predict()")
        self._model.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
            return self._model(X_t).numpy()


class SmallLSTMForecaster(_BaseRecurrentForecaster):
    """A small, CPU-friendly LSTM-based sequence forecaster."""

    _torch_cell_cls_name = "LSTM"


class SmallGRUForecaster(_BaseRecurrentForecaster):
    """A small, CPU-friendly GRU-based sequence forecaster."""

    _torch_cell_cls_name = "GRU"
