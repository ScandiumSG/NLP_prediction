from datasets import load_dataset

class DataFetcher():
    datasetName = ""
    datasetLoader = None
    downloadedData = []


    def downloadFullDataset(self):
        data = load_dataset("json", data_files=self.datasetName)
        return data["train"]
    
    def __init__(self):
        self.datasetLoader = localDatasets()
        # Default load all data
        self.datasetName = self.datasetLoader.getAllDatasets()
        self.downloadedData = self.downloadFullDataset()
    
    def getFullDataset(self):
        return self.downloadedData.shuffle()

    def getSmallDataset(self):
        subset = self.datasetLoader.getTestDataset()
        data = load_dataset("json", data_files=subset)
        data = data["train"]
        return data

    
# For reference:
#validFieldsAsString = "'meeting_date, 'data_split', 'sentence_id', 'sentence_order', 'speaker_id', 'speaker_name', 'sentence_text': 'disse søknadene foreslås behandlet straks og innvilget', 'sentence_language_code': 'nb-NO', 'text', 'start_time', 'end_time', 'normsentence_text', 'transsentence_text', 'translated', 'transcriber_id', 'reviewer_id', 'total_duration'"

class localDatasets():
    def __init__(self):
        pass

    def getAllDatasets(self):
        l1 = self.getEvalDataset()
        l2 = self.getTestDataset()
        l3 = self.getTrainDataset()
        list = l1 + l2 + l3
        return list

    def getEvalDataset(self):
        list = [
            "./data/eval/20170209.json", 
            "./data/eval/20180109.json", 
            "./data/eval/20180201.json", 
            "./data/eval/20180307.json", 
            "./data/eval/20180611.json"
        ]
        return list
    
    def getTestDataset(self):
        list = [
            "./data/test/20170207.json", 
            "./data/test/20171122.json", 
            "./data/test/20171219.json",
            "./data/test/20180530.json", 
        ]
        return list


    def getTrainDataset(self):
        list = [
            "./data/train/20170110.json", 
            "./data/train/20170208.json", 
            "./data/train/20170215.json", 
            "./data/train/20170216.json", 
            "./data/train/20170222.json", 
            "./data/train/20170314.json", 
            "./data/train/20170323.json", 
            "./data/train/20170403.json", 
            "./data/train/20170405.json", 
            "./data/train/20170419.json", 
            "./data/train/20170426.json", 
            "./data/train/20170503.json", 
            "./data/train/20170516.json", 
            "./data/train/20170613.json", 
            "./data/train/20170615.json", 
            "./data/train/20171007.json", 
            "./data/train/20171012.json", 
            "./data/train/20171018.json", 
            "./data/train/20171024.json", 
            "./data/train/20171208.json", 
            "./data/train/20171211.json", 
            "./data/train/20171213.json", 
            "./data/train/20180316.json", 
            "./data/train/20180321.json", 
            "./data/train/20180404.json", 
            "./data/train/20180410.json", 
            "./data/train/20180411.json", 
            "./data/train/20180601.json", 
            "./data/train/20180613.json", 
            "./data/train/20180615.json", 
        ]
        return list
    