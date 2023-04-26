from load import DataFetcher
from processData import DataProcessor
from PredictionModel import PredictionClassifier
from testModelPrediction import testPrediction

fetcher = DataFetcher()
processor = DataProcessor()

dataset = fetcher.getFullDataset()
dictionaryList = processor.jsonToDictionaryList(dataset)

partyList = processor.replacePersonWithParty(dictionaryList)

processor.makePartyDocumentGraph(partyList)

names = [t[0] for t in dictionaryList]
# Count the frequency of each name
name_count = {name: names.count(name) for name in set(names)}
# Sort the names by their frequency in descending order
sorted_names = sorted(name_count, key=name_count.get, reverse=True)
# Print the result
print("Sorted list of names by frequency:")
for name in sorted_names[0:15]:
    count = name_count[name]
    print(f"{name}: {count} time{'s' if count != 1 else ''}")