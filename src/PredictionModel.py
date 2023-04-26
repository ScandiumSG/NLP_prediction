from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class PredictionClassifier():
    vectorizer = None
    model = None
    model_trained = False
    # Trained parameter for quick access to eval
    X_test = None
    y_test = None

    def __init__(self, model="NB", vectorizer="Count", maxDF=0.8, forestCount=50):
        """
        Initalize the model

        Args: 
            model: Select the model to predict. Valid: "Forest", "NB", "Compliment". Defaults to "NB"
            vecorizer: Select the vectorizer to use. Valid choices: "tfidft", "count". Defaults to "Count"
            maxDF: The maximum document freqency before a word is removed when using the vectorizer. Defaults to 0.8.
            forestCount: number of estimaters to use when using the ForestClassifier. Defaults to 50 (overiding the 100 default)
        """

        # Set vectorizer
        if vectorizer.lower() == "tfidf":
            self.vectorizer = TfidfVectorizer(max_df=maxDF)
        elif vectorizer.lower() == "count":
            self.vectorizer = CountVectorizer(max_df=maxDF)
        else: # Default to Count
            print(f"Invalid vectorizer {vectorizer}, defaulting to CountVectorizer ('count')")
            self.vectorizer = CountVectorizer()
        # Set model
        if model.lower() == "forest":
            self.model = RandomForestClassifier(n_estimators=forestCount)
        elif model.lower() == "nb":
            self.model = MultinomialNB()
        elif model.lower() == "compliment":
            self.model = ComplementNB()
        else:
            print(f"Invalid model {model}, defaulting to MultinomialNB ('NB')")
            self.model = MultinomialNB()


    def getModel(self):
        if self.model_trained:
            return self.model
        else:
            return None

    def trainModel(self, dataset):
        # Use zip to split into 1 list of labels, and 1 list of sentences
        speakers, sentences = zip(*dataset)
        features = self.vectorizer.fit_transform(sentences)
        X_train, self.X_test, y_train, self.y_test = train_test_split(features, speakers, test_size=0.2)

        # Fit the model
        self.model.fit(X_train, y_train)
        self.model_trained = True

    def evaluateTrainedModel(self):
        if self.model_trained:
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)

            # Weighted f1 finds average weight for each label by number of true instances.
            # Alteration of "macro"
            f1 = f1_score(self.y_test, y_pred, average="weighted")
            print("Accuracy:", accuracy)
            print("F1 score: ", f1)
            return (accuracy, f1)
        else:
            print("Model is not trained yet!")
            return None
    
    def trainAndEvaluateModel(self, dataset):
        self.trainModel(dataset)
        self.evaluateTrainedModel()

    def generateConfusionMatrix(self, name):
        y_pred = self.model.predict(self.X_test)
        labels = ["R", "SV", "AP", "SP", "MDG", "KRF", "V", "H", "FRP", "other"]
        matrix = confusion_matrix(self.y_test, y_pred, labels=labels)

        
        matrixDisplay = ConfusionMatrixDisplay(matrix, display_labels=labels)
        matrixDisplay.plot(cmap=plt.cm.Blues)
        plt.savefig(f'images/{name}')
        plt.clf()

    def predictParty(self, sentence: str):
        features = self.vectorizer.transform([sentence])
        prediction = self.model.predict(features)
        return prediction[0]