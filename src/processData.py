from typing import List, Dict
from partyAffiliation import partiesForPolitications
import matplotlib.pyplot as plt
import numpy as np


class DataProcessor():
    parties = None

    def __init__(self):
        self.parties = partiesForPolitications

    def jsonToDictionaryList(self, dataset: List[Dict[str, any]]):
        print("Starting to translate json items to dictionary of (Speaker, Text)\n")
        returnList = []
        for data in dataset:
            itemPair = data.get("speaker_name"), data.get("text")
            returnList.append(itemPair)
        return returnList

    def replacePersonWithParty(self, dictionaryList: List[Dict[str, str]]):
        returnList = []
        requireSorting = []
        for name, sentence in dictionaryList:
            personParty = self.findPoliticalParty(name)
            if self.manualRemoval(name):
                pass
            elif personParty == None:
                requireSorting.append(name)
                returnList.append(("N/A", sentence))
            else:
                returnList.append((personParty, sentence))
        if len(requireSorting) != 0:
            print("Unrecognized people detected, the following was not categorized: ")
            print(*list(set(requireSorting)), sep="\n")
            print(f"Sorting required for {len(list(set(requireSorting)))} people.")
        return returnList

    def findPoliticalParty(self, personName: str): 
        for party, values in self.parties.items():
            if personName in values:
                return party
        return None

    def manualRemoval(self, personName: str):
        removed = ["unknown"]
        if personName in removed:
            return True
        else:
            return False
        
    def findVocabularMaxSize(self, listWithDictionaries: List[Dict[str, str]]):
        uniqueWords = set()
        for party, sent in listWithDictionaries:
            uniqueWords.update(sent.split())
        print("Max size: ", len(uniqueWords))

    def makePartyDocumentGraph(self, dataset):
        documents = np.array([])
        party_names = np.array([])
        for (party, values) in self.parties.items():
            party_names = np.append(party_names, party)
            number = int(len([s for p, s in dataset if p == party]))
            documents = np.append(documents, number)
        
        plt.bar(party_names, documents)
        plt.xlabel = "Party"
        plt.ylabel = "Documents"
        plt.show()