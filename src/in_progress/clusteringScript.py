from in_progress.clusteringModel import KMeansCluster
from load import DataFetcher
from processData import DataProcessor
from PredictionModel import PredictionClassifier

fetcher = DataFetcher()
processor = DataProcessor()
dataset = fetcher.getFullDataset()
dictionaryList = processor.jsonToDictionaryList(dataset)
partyList = processor.replacePersonWithParty(dictionaryList)

cluster = KMeansCluster()
cluster.trainAndEvaluateModel(partyList)

cluster.visualizeCluster()