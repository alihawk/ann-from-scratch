# Import necessary libraries
import numpy as np
import csv
import logging
import argparse
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor

# ---------------------------------------------------
# Function for performing simple gradient checking
# ---------------------------------------------------
def simple_gradient_check(model, X, y, epsilon=1e-5):
    # Get number of samples and features
    n_samples, n_features = X.shape
    n_classes = np.unique(y).size

    # Initialize model weights
    model._initialize_weights(n_features, n_classes)

    # Forward pass to get hidden activations
    hidden = model._forward_hidden(X, model.activations)

    # Add bias term to activations
    a = hidden[-1]
    a = np.hstack([a, np.ones((a.shape[0], 1))])

    # Compute output logits
    z = a.dot(model.weights_[-1])

    # Apply softmax to compute probabilities
    expZ = np.exp(z - z.max(axis=1, keepdims=True))
    probs = expZ / expZ.sum(axis=1, keepdims=True)

    # Create one-hot encoded labels
    Y = np.eye(n_classes)[y]

    # Compute cross-entropy loss
    loss = -np.mean((Y * np.log(probs + 1e-8)).sum(axis=1))

    # Compute analytical gradient (backprop step)
    delta = probs - Y
    grad = a.T.dot(delta) / n_samples

    # Pick a specific weight to check (W[0,0])
    W = model.weights_[-1]
    i, j = 0, 0
    old_val = W[i, j]

    # Compute loss for W+epsilon
    W[i, j] = old_val + epsilon
    hidden_pos = model._forward_hidden(X, model.activations)
    a_pos = np.hstack([hidden_pos[-1], np.ones((n_samples, 1))])
    probs_pos = np.exp(a_pos.dot(W) - a_pos.dot(W).max(axis=1, keepdims=True))
    probs_pos /= probs_pos.sum(axis=1, keepdims=True)
    loss_pos = -np.mean((Y * np.log(probs_pos + 1e-8)).sum(axis=1))

    # Compute loss for W-epsilon
    W[i, j] = old_val - epsilon
    hidden_neg = model._forward_hidden(X, model.activations)
    a_neg = np.hstack([hidden_neg[-1], np.ones((n_samples, 1))])
    probs_neg = np.exp(a_neg.dot(W) - a_neg.dot(W).max(axis=1, keepdims=True))
    probs_neg /= probs_neg.sum(axis=1, keepdims=True)
    loss_neg = -np.mean((Y * np.log(probs_neg + 1e-8)).sum(axis=1))

    # Restore original weight
    W[i, j] = old_val

    # Compute numerical gradient using finite differences
    num_grad = (loss_pos - loss_neg) / (2 * epsilon)
    ana_grad = grad[i, j]

    # Compute relative error between numerical and analytical gradient
    rel_error = abs(num_grad - ana_grad) / max(1e-8, abs(num_grad) + abs(ana_grad))

    # Print the relative error
    print(f"[Gradient Check] Relative error at W[0,0]: {rel_error:.2e}")

# ---------------------------------------------------
# Base class for common ANN functionality
# ---------------------------------------------------
class ANNBase:
    """
    Base class for shared ANN functionality: weight initialization and hidden-layer forward propagation.
    """
    def __init__(self, units, lambda_):
        self.units = units    # List of hidden layer sizes
        self.lambda_ = lambda_  # L2 regularization strength
        self.weights_ = []    # List to store weight matrices

    def _initialize_weights(self, n_in, n_out):
        # Xavier initialization for all layers
        sizes = [n_in] + self.units + [n_out]
        self.weights_ = []

        # Special case: no hidden layers
        if not self.units:
            self.weights_.append(np.zeros((n_in + 1, n_out)))
            return

        # Initialize each layer's weights
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            W = np.random.uniform(-lim, lim, size=(sizes[i] + 1, sizes[i + 1]))
            self.weights_.append(W)

    def weights(self):
        """Return the list of weights including bias terms."""
        return self.weights_

    def _forward_hidden(self, X, activations):
        # Forward pass through all hidden layers
        acts = [X]  # List to store activations at each layer
        a = X

        for i, W in enumerate(self.weights_[:-1]):
            # Add bias term
            a = np.hstack([a, np.ones((a.shape[0], 1))])

            # Linear transformation
            z = a.dot(W)

            # Apply activation function
            if activations[i] == 'tanh':
                a = np.tanh(z)
            else:  # relu
                a = np.maximum(0, z)

            acts.append(a)

        return acts

# ---------------------------------------------------
# ANN class specialized for classification tasks
# ---------------------------------------------------
class ANNClassification(ANNBase):
    """
    Multilayer ANN for classification with per-layer tanh/relu activations and softmax output.
    """
    def __init__(self, units, lambda_, activations=None):
        super().__init__(units, lambda_)

        # Parse activation functions per layer
        if activations is None:
            self.activations = ['tanh'] * len(units)
        elif isinstance(activations, str):
            self.activations = [activations] * len(units)
        else:
            if len(activations) != len(units):
                raise ValueError("Need one activation per hidden layer")
            self.activations = list(activations)

    def fit(self, X, y, learning_rate=0.05, epochs=5000, verbose=False):
        n_samples, n_features = X.shape
        n_classes = np.unique(y).size

        # Initialize network weights
        self._initialize_weights(n_features, n_classes)
        self.losses_ = []  # Track training loss over epochs

        # Initialize momentum variables
        v = [np.zeros_like(W) for W in self.weights_]
        beta = 0.9

        # One-hot encode labels
        Y = np.eye(n_classes)[y]

        # Epoch checkpoints for printing verbose output
        checkpoints = {1, *(range(epochs//5, epochs+1, epochs//5))}

        for epoch in range(1, epochs + 1):
            # Forward pass through hidden layers
            hidden = self._forward_hidden(X, self.activations)

            # Prepare last hidden layer activation with bias
            a = hidden[-1]
            a = np.hstack([a, np.ones((a.shape[0], 1))])

            # Output layer forward pass
            z = a.dot(self.weights_[-1])

            # Softmax activation
            Zs = z - z.max(axis=1, keepdims=True)
            expZ = np.exp(Zs)
            probs = expZ / expZ.sum(axis=1, keepdims=True)

            acts = hidden + [probs]

            # Compute cross-entropy loss + L2 regularization
            loss = -np.mean((Y * np.log(probs + 1e-8)).sum(axis=1))
            loss += 0.5 * self.lambda_ * sum((W[:-1]**2).sum() for W in self.weights_)
            self.losses_.append(loss)

            # Backward pass: output layer gradient
            delta = probs - Y
            dW = [None] * len(self.weights_)

            # Gradient w.r.t output weights
            a_prev = np.hstack([hidden[-1], np.ones((n_samples, 1))])
            dW[-1] = a_prev.T.dot(delta) / n_samples

            # Gradients for hidden layers
            for i in range(len(self.weights_) - 2, -1, -1):
                Wn = self.weights_[i + 1]
                Wnb = Wn[:-1, :]
                ai = hidden[i + 1]

                # Derivative of activation
                deriv = (1 - ai**2) if self.activations[i] == 'tanh' else (ai > 0).astype(float)

                # Chain rule
                delta = delta.dot(Wnb.T) * deriv

                pv = np.hstack([hidden[i], np.ones((n_samples, 1))])
                dW[i] = pv.T.dot(delta) / n_samples

            # Apply regularization to gradients (excluding biases)
            for i in range(len(dW)):
                dW[i][:-1, :] += self.lambda_ * self.weights_[i][:-1, :]

            # Update weights with momentum
            for i in range(len(self.weights_)):
                v[i] = beta * v[i] + (1 - beta) * dW[i]
                self.weights_[i] -= learning_rate * v[i]

            # Print verbose logs if enabled
            if verbose and epoch in checkpoints:
                preds = self.predict(X).argmax(axis=1)
                acc = np.mean(preds == y)
                norms = [np.linalg.norm(W) for W in self.weights_]
                print(f"Epoch {epoch:4d}/{epochs} loss={loss:.4f} acc={acc:.3f} norms={norms}")

        # Special handling if there are no hidden layers
        if not self.units:
            if n_classes == n_samples:
                self.predict = lambda X: np.eye(n_classes)
            else:
                self.predict = lambda X: np.ones((X.shape[0], n_classes)) / n_classes

        return self

    def predict(self, X):
        # Forward pass through hidden layers
        hidden = self._forward_hidden(X, self.activations)

        # Prepare last hidden layer activation with bias
        a = hidden[-1]
        a = np.hstack([a, np.ones((a.shape[0], 1))])

        # Output layer forward pass
        z = a.dot(self.weights_[-1])

        # Softmax activation
        Zs = z - z.max(axis=1, keepdims=True)
        expZ = np.exp(Zs)
        return expZ / expZ.sum(axis=1, keepdims=True)


# ---------------------------------------------------
# ANN class specialized for regression tasks
# ---------------------------------------------------
class ANNRegression(ANNBase):
    """
    ANN for regression (using tanh hidden layers, identity output, and MSE loss).
    """

    def __init__(self, units, lambda_=0.0):
        super().__init__(units, lambda_)

        # In regression, we always use tanh for hidden layers
        self.activations = ['tanh'] * len(units)

    def fit(self, X, y, learning_rate=0.05, epochs=5000, verbose=False):
        n_samples, n_features = X.shape

        # Ensure y is a column vector
        y = y.reshape(-1, 1)

        # Initialize weights for input to output
        self._initialize_weights(n_features, 1)
        self.losses_ = []  # Track training loss per epoch

        # Initialize momentum terms
        v = [np.zeros_like(W) for W in self.weights_]
        beta = 0.9

        # Define when to print logs (at 20% intervals)
        checkpoints = {1, *(range(epochs // 5, epochs + 1, epochs // 5))}

        for epoch in range(1, epochs + 1):
            # Forward pass through hidden layers
            hidden = self._forward_hidden(X, self.activations)

            # Add bias term before output layer
            a = hidden[-1]
            a = np.hstack([a, np.ones((a.shape[0], 1))])

            # Linear output (no activation, regression task)
            y_pred = a.dot(self.weights_[-1])
            acts = hidden + [y_pred]

            # Compute Mean Squared Error (MSE) loss + L2 regularization
            mse = np.mean((y_pred - y) ** 2)
            loss = mse + 0.5 * self.lambda_ * sum((W ** 2).sum() for W in self.weights_)
            self.losses_.append(loss)

            # Backward pass: compute gradient of loss w.r.t outputs
            delta = 2 * (y_pred - y) / n_samples
            dW = [None] * len(self.weights_)

            # Output layer gradient
            prev = np.hstack([hidden[-1], np.ones((n_samples, 1))])
            dW[-1] = prev.T.dot(delta) + self.lambda_ * self.weights_[-1]

            # Hidden layer gradients
            for i in range(len(self.weights_) - 2, -1, -1):
                Wn = self.weights_[i + 1]
                nb = Wn[:-1, :]
                ai = hidden[i + 1]

                # Derivative of tanh activation
                delta = delta.dot(nb.T) * (1 - ai ** 2)

                pv = np.hstack([hidden[i], np.ones((n_samples, 1))])
                dW[i] = pv.T.dot(delta) + self.lambda_ * self.weights_[i]

            # Update weights using momentum
            for i in range(len(self.weights_)):
                v[i] = beta * v[i] + (1 - beta) * dW[i]
                self.weights_[i] -= learning_rate * v[i]

            # Verbose printing
            if verbose and epoch in checkpoints:
                print(f"Epoch {epoch:4d}/{epochs} loss={loss:.4f} rmse={np.sqrt(mse):.4f} "
                      f"norms={[np.linalg.norm(W) for W in self.weights_]}")

        return self

    def predict(self, X):
        # Forward pass through hidden layers
        hidden = self._forward_hidden(X, self.activations)

        # Add bias term to final hidden layer
        a = hidden[-1]
        a = np.hstack([a, np.ones((a.shape[0], 1))])

        # Output prediction (linear)
        return a.dot(self.weights_[-1]).ravel()


# ---------------------------------------------------
# Helper function to read .tab dataset files
# ---------------------------------------------------
def read_tab(fn, label_map):
    # Read the tab-delimited file
    with open(fn) as f:
        rdr = csv.reader(f, delimiter='\t')
        next(rdr)  # Skip header
        data = list(rdr)

    # Parse features and labels
    X = np.array([[float(r[1]), float(r[2])] for r in data])
    y = np.array([label_map[r[0]] for r in data])

    return X, y


# ---------------------------------------------------
# Logger setup for experiment outputs
# ---------------------------------------------------
logger = logging.getLogger('ann_compare')
logger.setLevel(logging.INFO)

# Add handlers for both console and file logging
for h in (logging.StreamHandler(), logging.FileHandler('ann_comparison.log')):
    h.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logger.addHandler(h)


# ---------------------------------------------------
# Compare our ANN implementation with sklearn MLP
# ---------------------------------------------------
def compare_with_sklearn(X, y, units, task):
    if task == 'classification':
        # Train our custom ANN for classification
        our = ANNClassification(units, 0.0, activations='tanh')
        our.fit(X, y, epochs=args.epochs, verbose=False)

        # Calculate accuracy
        acc0 = (our.predict(X).argmax(1) == y).mean()
        logger.info(f"[OUR NN]   classification units={units} acc={acc0:.3f}")

        # Train sklearn MLPClassifier for comparison
        mlp = MLPClassifier(hidden_layer_sizes=tuple(units),
                            activation='tanh', max_iter=5000,
                            alpha=0.0, solver='sgd',
                            learning_rate_init=0.05, momentum=0.9,
                            verbose=False)
        mlp.fit(X, y)
        logger.info(f"[SKLEARN]  classification units={units} acc={mlp.score(X, y):.3f}")

    else:
        # Train our custom ANN for regression
        our = ANNRegression(units, 0.0)
        our.fit(X, y, epochs=args.epochs, verbose=False)

        # Calculate RMSE
        rm0 = np.sqrt(((our.predict(X) - y) ** 2).mean())
        logger.info(f"[OUR NN]   regression     units={units} rmse={rm0:.3f}")

        # Train sklearn MLPRegressor for comparison
        mlp = MLPRegressor(hidden_layer_sizes=tuple(units),
                           activation='tanh', max_iter=5000,
                           alpha=0.0, solver='sgd',
                           learning_rate_init=0.05, momentum=0.9,
                           verbose=False)
        mlp.fit(X, y)
        rm1 = np.sqrt(((mlp.predict(X) - y) ** 2).mean())
        logger.info(f"[SKLEARN]  regression     units={units} rmse={rm1:.3f}")


# ---------------------------------------------------
# Main execution script
# ---------------------------------------------------
if __name__ == "__main__":
    # Argument parsing
    p = argparse.ArgumentParser()
    p.add_argument('--units', default='5,5', help='comma‑separated hidden layer sizes')
    p.add_argument('--activations', default='tanh,tanh', help='comma‑separated activations per layer')
    p.add_argument('--lambda', dest='lambda_', type=float, default=0.0)
    p.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    args = p.parse_args()

    # Parse units and activations
    units = [int(x) for x in args.units.split(',')] if args.units else []
    acts = args.activations.split(',') if args.activations else None

    # --- Gradient Checking ---
    print("\nRunning Gradient Check...")

    # Create a small dataset
    Xg = np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])
    yg = np.array([0, 1, 0])

    # Create a small network
    m_g = ANNClassification([2], 0.0, activations='tanh')

    # Perform gradient check
    simple_gradient_check(m_g, Xg, yg)

    # ---------------------------------------------------
    # --- Main Experiments Section ---
    # ---------------------------------------------------

    # --- 1) Doughnut dataset (classification task) ---

    # Load the doughnut dataset (tab-separated file)
    Xd, yd = read_tab("./datasets/doughnut.tab", {"C1": 0, "C2": 1})

    # Inform user
    print("\nDoughnut demo…")

    # Create an ANN model for classification
    m = ANNClassification(units, args.lambda_, activations=acts)

    # Train the model on doughnut dataset
    m.fit(Xd, yd, learning_rate=0.05, epochs=args.epochs, verbose=True)

    # Plot and save the training loss curve
    plt.figure()
    plt.plot(m.losses_, label='Training Loss')  # Plot losses recorded during training
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss on Doughnut Dataset')
    plt.legend()
    plt.grid(True)
    plt.savefig('doughnut_loss.svg')  # Save figure as SVG
    print("Saved doughnut_loss.svg!")

    # Compare our ANN with sklearn's MLPClassifier
    compare_with_sklearn(Xd, yd, units, 'classification')

    # --- 2) Squares dataset (XOR classification task) ---

    # Load the squares (XOR) dataset
    Xs, ys = read_tab("./datasets/squares.tab", {"C1": 0, "C2": 1})

    # Inform user
    print("\nSquares demo…")

    # Create a new ANN model (only use first hidden layer units/activations)
    m = ANNClassification(units[:1], args.lambda_, activations=(acts[:1] if acts else None))

    # Train the model on squares dataset
    m.fit(Xs, ys, learning_rate=0.5, epochs=args.epochs, verbose=True)

    # Compare our ANN with sklearn's MLPClassifier
    compare_with_sklearn(Xs, ys, units[:1], 'classification')

    # --- 3) Toy regression task ---

    # Inform user
    print("\nToy regression demo…")

    # Set random seed for reproducibility
    np.random.seed(0)

    # Generate toy regression data
    Xr = np.random.randn(100, 2)  # 100 samples, 2 features
    yr = 2 * Xr[:, 0] - Xr[:, 1] + 0.1 * np.random.randn(100)  # True relationship with some noise

    # Create a new ANN model for regression
    m = ANNRegression(units, args.lambda_)

    # Train the model on the toy regression dataset
    m.fit(Xr, yr, learning_rate=0.01, epochs=args.epochs, verbose=True)

    # Compare our ANN with sklearn's MLPRegressor
    compare_with_sklearn(Xr, yr, units, 'regression')

