import numpy as np
from sklearn.ensemble import IsolationForest


def test_model():
    X = np.random.rand(50, 2)

    model = IsolationForest()

    model.fit(X)

    preds = model.predict(X)

    assert len(preds) == 50
