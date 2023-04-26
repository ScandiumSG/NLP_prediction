from load import DataFetcher
from processData import DataProcessor
from PredictionModel import PredictionClassifier
from testModelPrediction import testPrediction

fetcher = DataFetcher()
processor = DataProcessor()

dataset = fetcher.getFullDataset()
dictionaryList = processor.jsonToDictionaryList(dataset)

models = []

print("(Count-NB) Evaluating individual speaker accuracy: ")
speaker_model = PredictionClassifier(model="nb", vectorizer="count")
speaker_model.trainAndEvaluateModel(dictionaryList)
models.append(("(Count-NB - Person)", speaker_model))


print("\n(Count-NB) Evaluation political party accuracy: ")
partyList = processor.replacePersonWithParty(dictionaryList)
party_model = PredictionClassifier(model="nb", vectorizer="count")
party_model.trainAndEvaluateModel(partyList)
models.append(("(Count-NB)", party_model))

print("\n(TF_IDF-NB) Evaluating political party accuracy: ")
tfidfModel = PredictionClassifier(model="nb", vectorizer="tfidf")
tfidfModel.trainAndEvaluateModel(partyList)
models.append(("(TF_IDF-NB)", tfidfModel))

print("(Count-Compliment) Evaluating political party accuracy: ")
complimentNB2 = PredictionClassifier(model="compliment", vectorizer="count")
complimentNB2.trainAndEvaluateModel(partyList)
models.append(("(Count-Compliment)", complimentNB2))


print("\n(TF_IDF-Compliment) Evaluating political party accuracy: ")
complimentNB = PredictionClassifier(model="compliment", vectorizer="tfidf")
complimentNB.trainAndEvaluateModel(partyList)
models.append(("(TF_IDF-Compliment)", complimentNB))

enableForest = False
if enableForest:
    # Disable forest if you dont have 10min to generate...
    print("\n(Count-Forest) Evaluating political party accuracy: ")
    forest_count = PredictionClassifier(model="forest", vectorizer="count")
    forest_count.trainAndEvaluateModel(partyList)
    models.append(("(Count-Forest)", forest_count))

    print("\n(TF_IDF-Forest) Evaluating political party accuracy: ")
    forest_tfidf = PredictionClassifier(model="forest", vectorizer="tfidf")
    forest_tfidf.trainAndEvaluateModel(partyList)
    models.append(("(TF_IDF-Forest)", forest_tfidf))

