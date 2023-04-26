from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class KMeansCluster():
    model = None
    model_trained = False
    vectorizer = None
    dataset = None
    # Trained parameter for quick access to eval
    X_test = None
    y_test = None

    def __init__(self, Nclusters=10, RandomStates=32):
        self.model = KMeans(n_init=10, n_clusters=Nclusters, random_state=RandomStates)
        self.vectorizer = CountVectorizer()

    def trainModel(self, dataset, testSize=0.2):
        self.dataset = dataset
        # Use zip to split into 1 list of labels, and 1 list of sentences
        party, sentences = zip(*dataset)
        self.model_trained = True
        self.sentences = sentences
        # Make training/testing sets
        sentences_train, sentences_test, parties_train, self.y_test = train_test_split(sentences, party, test_size=testSize, random_state=42)

        X_train = self.vectorizer.fit_transform(sentences_train)
        self.X_test = self.vectorizer.transform(sentences_test)

        # Train/Fit model
        self.model.fit(X_train)

    def evaluateTrainedModel(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Accuracy: {accuracy}")

    def trainAndEvaluateModel(self, dataset):
        self.trainModel(dataset)
        self.evaluateTrainedModel()

    def visualizeCluster(self):
        data = self.dataset
        party, sentences = zip(*data)
        num_parts = 10
        parts = [sentences[i:i+(len(sentences)//num_parts)] for i in range(0, len(sentences), len(sentences)//num_parts)]
        for part in parts[0:9]:
            X = self.vectorizer.fit_transform(part)

            # Perform PCA to reduce the dimensionality of the data to 2
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X.toarray())

            # Train a clustering model using KMeans
            kmeans = KMeans(n_init=10, n_clusters=10, random_state=42)
            kmeans.fit(X)

            # Plot the clustering results
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_)
            plt.savefig(f'kmeans_batch.png')
        plt.title("KMeans Clustering Results")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.show()