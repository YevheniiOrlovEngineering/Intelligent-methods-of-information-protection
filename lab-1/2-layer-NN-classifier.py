import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import load_data


sns.set(style="whitegrid", palette="muted", font_scale=1.5)


def plot_person(x_set: np.ndarray, y_set: np.ndarray, idx: int) -> None:
    img = x_set.T[idx].reshape(64, 64)
    plt.imshow(img, cmap="Greys",  interpolation="nearest")
    plt.title("true label: %d" % y_set.T[idx])
    plt.show()


class Sigmoid:
    def __call__(self, z):
        """
        Computes the sigmoid of z

        Arguments:
        z -- scalar or numpy array of any size.

        Return:
        sigmoid(z)
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def prime(z):
        """
        Computes the derivative of sigmoid of z

        Arguments:
        z -- scalar or numpy array of any size.

        Return:
        Sigmoid prime
        """
        return (1 / (1 + np.exp(-z))) * (1 - (1 / (1 + np.exp(-z))))
        

def one_hot(Y, n_classes):
    """
    Encode labels into a one-hot representation

    Arguments:
    Y -- array of input labels of shape (1, n_samples)
    n_classes -- number of classes

    Returns:
    onehot, a matrix of labels by samples. For each column, the ith index will be
        "hot", or 1, to represent that index being the label; shape - (n_classes, n_samples)
    """

    y_matrix = np.zeros((Y.shape[1], n_classes))
    for row, label in zip(y_matrix, Y[0]):
        row[label] = 1

    return y_matrix.T
    

def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost

    Arguments:
    A2 -- sigmoid output of the hidden layer activation, of shape (classes, n_examples)
    Y -- labels of shape (classes, n_examples)

    Returns:
    cost -- cross-entropy cost equation (4)
    """

    m = Y.shape[1]  # number of examples

    # Compute the cross-entropy cost
    
    cost = 0
    for a_cl, y_cl in zip(A2.T, Y.T):
        for a_val, y_val in zip(a_cl, y_cl):
            cost += (y_val * np.log(a_val)) + (1 - y_val) * (np.log(1 - a_val))
    cost = (-1/m) * cost

    return cost


class Regularization:
    """
    Regularization class

    Arguments:
    lambda_1 -- regularization coefficient for l1 regularization
    lambda_2 -- regularization coefficient for l2 regularization
    """

    def __init__(self, lambda_1, lambda_2):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

    def l1(self, W1, W2, m):
        """
        Compute l1 regularization part

        Arguments:
        W1 -- weights of shape (n_hidden_units, n_features)
        W2 -- weights of shape (output_size, n_hidden_units)
        m -- n_examples

        Returns:
        l1_term -- float
        """
        
        return (self.lambda_1 / m) * (np.sum(np.abs(W1) + np.abs(W2)))

    def l1_grad(self, W1, W2, m):
        """
        Compute l1 regularization term

        Arguments:
        W1 -- weights of shape (n_hidden_units, n_features)
        W2 -- weights of shape (output_size, n_hidden_units)
        m -- n_examples

        Returns:
         dict with l1_grads "dW1" and "dW2"
            which are grads by corresponding weights
        """
        
        return {"dW1": self.lambda_1 / m * np.sign(W1),
                "dW2": self.lambda_1 / m * np.sign(W2)}

    def l2(self, W1, W2, m):
        """
        Compute l2 regularization term

        Arguments:
        W1 -- weights of shape (n_hidden_units, n_features)
        W2 -- weights of shape (output_size, n_hidden_units)
        m -- n_examples

        Returns:
        l2_term: float
        """
        
        l2_term = (self.lambda_2 / (2*m)) * (np.linalg.norm(W1) ** 2 + np.linalg.norm(W2) ** 2)
        return l2_term

    def l2_grad(self, W1, W2, m):
        """
        Compute l2 regularization term

        Arguments:
        W1 -- weights of shape (n_hidden_units, n_features)
        W2 -- weights of shape (output_size, n_hidden_units)
        m -- n_examples

        Returns:
        l2_grads: dict with keys "dW1" and "dW2"
        """
        
        return {"dW1": self.lambda_2 / m * W1,
                "dW2": self.lambda_2 / m * W2}
        

class NeuralNetwork:
    """
    Arguments:
    n_features: int -- Number of features
    n_hidden_units: int -- Number of hidden units
    n_classes: int -- Number of classes
    learning_rate: float
    reg: instance of Regularization class
    """

    def __init__(self, n_features,
                 n_hidden_units,
                 n_classes,
                 learning_rate,
                 reg=Regularization(0.1, 0.2),
                 sigm=Sigmoid()):

        self.n_features = n_features
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.n_hidden_units = n_hidden_units
        self.reg = reg
        self.sigm = sigm
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.initialize_parameters()

    def initialize_parameters(self):
        """
        W1 -- weight matrix of shape (self.n_hidden_units, self.n_features)
        b1 -- bias vector of shape (self.n_hidden_units, 1)
        W2 -- weight matrix of shape (self.n_classes, self.n_hidden_units)
        b2 -- bias vector of shape (self.n_classes, 1)
        """
        np.random.seed(42)

        self.W1 = np.random.normal(0, 0.01, (self.n_hidden_units, self.n_features))
        self.W2 = np.random.normal(0, 0.01, (self.n_classes, self.n_hidden_units))
        self.b1 = np.zeros((self.n_hidden_units, 1))
        self.b2 = np.zeros((self.n_classes, 1))

    def forward_propagation(self, X):
        """
        Arguments:
        X -- input data of shape (number of features, number of examples)

        Returns:
        dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Forward Propagation to calculate A2 (probabilities)
        
        Z1 = self.W1 @ X + self.b1
        A1 = self.sigm(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = self.sigm(Z2)

        return {
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2
        }

    def backward_propagation(self, X, Y, cache):
        """
        Arguments:
        X -- input data of shape (number of features, number of examples)
        Y -- one-hot encoded vector of labels with shape (n_classes, n_samples)
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"

        Returns:
        dictionary containing gradients "dW1", "db1", "dW2", "db2"
        """
        m = X.shape[1]

        A1 = cache["A1"]
        A2 = cache["A2"]

        # Backward propagation: calculate dW1, db1, dW2, db2.
        
        dZ2 = A2 - Y
        dZ1 = self.W2.T @ dZ2 * self.sigm.prime(cache["Z1"])

        dW2 = (1/m) * dZ2 @ A1.T + \
              self.reg.l1_grad(self.W1, self.W2, m)["dW2"] + \
              self.reg.l2_grad(self.W1, self.W2, m)["dW2"]

        dW1 = (1/m) * dZ1 @ X.T + \
              self.reg.l1_grad(self.W1, self.W2, m)["dW1"] + \
              self.reg.l2_grad(self.W1, self.W2, m)["dW1"]

        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

        return {
            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2
        }

    def update_parameters(self, grads):
        """
        Updating parameters using the gradient descent update rule

        Arguments:
        grads -- python dictionary containing gradients "dW1", "db1", "dW2", "db2"
        """
        # Retrieving each gradient from the dictionary "grads"

        dW1 = grads["dW1"]
        dW2 = grads["dW2"]
        db1 = grads["db1"]
        db2 = grads["db2"]

        # Updating each parameter
        
        self.W1 = self.W1 - self.learning_rate * dW1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b1 = self.b1 - self.learning_rate * db1
        self.b2 = self.b2 - self.learning_rate * db2


class NNClassifier:
    """
    NNClassifier class

    Arguments:
    model -- instance of NN
    epochs: int -- Number of epochs
    """

    def __init__(self, model, epochs=1000):
        self.model = model
        self.epochs = epochs
        self._cost = []  # WList of values of cost function after each epoch to build graph later

    def fit(self, X, Y) -> None:
        """
        Learn weights and errors from training data

        Arguments:
        X -- input data of shape (number of features, number of examples)
        Y -- labels of shape (1, number of examples)
        """

        Y = one_hot(Y, np.max(Y) + 1)

        for _ in range(int(self.epochs / 10)):
            act_and_inp_val = self.model.forward_propagation(X)
            self._cost.append(compute_cost(act_and_inp_val["A2"], Y))
            weights_derivatives = self.model.backward_propagation(X, Y, act_and_inp_val)
            self.model.update_parameters(weights_derivatives)

    def predict(self, X):
        """
        Generate array of predicted labels for the input dataset

        Arguments:
        X -- input data of shape (number of features, number of examples)

        Returns:
        predicted labels of shape (1, n_samples)
        """

        cache = self.model.forward_propagation(X)

        return np.argmax(cache["A2"], axis=0).T


def accuracy(pred: np.ndarray, labels: np.ndarray) -> np.float64:
    return (np.sum(pred == labels, axis=1) / float(labels.shape[1]))[0]


def plot_error(model: NNClassifier, epochs: int) -> None:
    plt.plot(range(len(model._cost)), model._cost)
    plt.ylim([0, epochs])
    plt.ylabel("Error")
    plt.xlabel("Epochs")
    plt.show()


train_set_x, test_set_x, train_set_y, test_set_y = load_data()

NN = NeuralNetwork(n_features=784, n_hidden_units=30, n_classes=10, learning_rate=0.01)
classifier = NNClassifier(model=NN, epochs=5000)

classifier.fit(train_set_x, train_set_y)

plot_error(classifier, 10)

pred_train = classifier.predict(train_set_x)
pred_test = classifier.predict(test_set_x)

print("train set accuracy: ", accuracy(pred_train, train_set_y))
print("test set accuracy: ", accuracy(pred_test, test_set_y))

