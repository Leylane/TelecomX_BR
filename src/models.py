# Modeling functions

from sklearn.linear_model import LogisticRegression

def train_logistic(X, y):
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    return model
