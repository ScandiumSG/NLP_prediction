from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import tensorflow as tf

class CNNModel():
    vectorizer = None
    model = None
    model_trained = False
    # Trained parameter for quick access to eval
    X_test = None
    y_test = None

    def __init__(self):
        # define the CNN model
        self.model = tf.keras.models.Sequential([
            # 45000ish is size of vocabulary
            tf.keras.layers.Embedding(input_dim=4500, output_dim=32, input_length=1000),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        # compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def getModel(self):
        if self.model_trained:
            return self.model
        else:
            return None

    def trainModel(self, dataset):
        # Use zip to split into 1 list of labels, and 1 list of sentences
        speakers, sentences = zip(*dataset[0:20000])
        vectorizer = CountVectorizer()
        features = vectorizer.fit_transform(sentences)

        # TODO: CANNOT DO THIS AS I DONT HAVE 20GiB to make this array. 
        # Putting CNN model away due to technical limitation
        features = features.toarray()
        X_train, self.X_test, y_train, self.y_test = train_test_split(features, speakers, test_size=0.2)

        # Fit the model
        self.model.fit(X_train, y_train, validation_data=(self.X_test, self.y_test), epochs=10, batch_size=32)
        self.model_trained = True

    def evaluateTrainedModel(self):
        if self.model_trained:
            loss, binary_accuracy, precision, recall, auc = self.model.evaluate(self.x_test, self.y_test)

            f1 = 2 * (precision * recall) / (precision + recall)
            print("Accuracy:", binary_accuracy)
            print("F1 score: ", f1)
            return (binary_accuracy, f1)
        else:
            return None
    
    def trainAndEvaluateModel(self, dataset):
        self.trainModel(dataset)
        self.evaluateTrainedModel()
