import numpy as np
from sklearn.ensemble import IsolationForest

def test_model():

    X = np.random.rand(100,5)

    model = IsolationForest()

    model.fit(X)

    preds = model.predict(X)

    assert len(preds) == 100