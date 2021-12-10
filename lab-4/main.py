from dataset import get_faces
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from time import time
from pandas import DataFrame
from view import display_table

if __name__ == "__main__":

    X_train, y_train, X_test, y_test = get_faces()
    model = Sequential()

    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    start = time()
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
    stop = time()

    results = DataFrame([[model.evaluate(X_train, y_train)[-1],
                          model.evaluate(X_test, y_test)[-1],
                          stop - start]],
                        columns=("Training Accuracy", "Testing Accuracy", "Execution Time"))

    display_table(results)
