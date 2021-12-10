import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from dataset import load_data


def accuracy(pred: np.ndarray, labels: np.ndarray) -> np.float64:
    return (np.sum(pred == labels, axis=1) / float(labels.shape[1]))[0]


def plot_person(x_set: np.ndarray, y_set: np.ndarray, idx: int) -> None:
    img = x_set.T[idx].reshape(64, 64)
    plt.imshow(img, cmap="Greys",  interpolation="nearest")
    plt.title("true label: %d" % y_set.T[idx])
    plt.show()


if __name__ == "__main__":

    train_set_x, test_set_x, train_set_y, test_set_y = load_data(ex_num=20)

    clf = MLPClassifier(solver="sgd", alpha=1e-5,
                        random_state=1, max_iter=5000,
                        hidden_layer_sizes=(100, 100))

    clf.fit(X=list(train_set_x.T), y=list(np.ndarray.flatten(train_set_y)))

    pred_train = clf.predict(train_set_x.T)
    pred_test = clf.predict(test_set_x.T)

    print("train set accuracy: ", accuracy(pred_train, train_set_y))
    print("test set accuracy: ",  accuracy(pred_test, test_set_y))

