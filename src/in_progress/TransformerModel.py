from typing import List
import tensorflow as tf
from transformers import BertTokenizerFast, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers import AutoTokenizer, TFAutoModel, TFAutoModelForSequenceClassification
import numpy as np

class BertModel: 
    model = None
    tokenizer: PreTrainedTokenizer = None

    def __init__(self):
        pretrainedModel = "NbAiLab/nb-bert-base"
        self.model = TFAutoModel.from_pretrained(pretrainedModel)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrainedModel)

    # Non functional
    def trainModel(self, dataset, learningRate=3e-5):
        speakers, sentences = zip(*dataset)
        trainingSetEncoded = self.tokenizer.batch_encode_plus(
            sentences,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='np'
        )
        # Invalid tensor rank, need rank >1. 
        trainingDataset = tf.data.Dataset.from_tensor_slices((trainingSetEncoded, speakers)).shuffle(50)

        # Config optimizer and some metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        self.model.fit(trainingDataset)

    # TODO: Fix to insert labels to our dataset.
    # Have said it before, and i'll say it again. Python developers are literally the worst developers ever, past present and future. 
    # Their idea of extensive documentation is trivial at best, code-examples? What are those? 
    # This model would have been cool to test, but i literally cannot for the life of me figure out how to get this tf_dataset to also have labels....
    # /RantOver
    def trainModel2(self, dataset, learningRate=3e-5):
        def tokenize_dataset(data):
        # Keys of the returned dictionary will be added to the dataset as columns
            return self.tokenizer(data["text"], )
        dataset = dataset.map(tokenize_dataset)
        labels = np.array(dataset["speaker_name"])
        tf_dataset = self.model.prepare_tf_dataset(dataset, batch_size=16, shuffle=True, tokenizer=self.tokenizer)
        # Config optimizer and some metrics
        optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

        self.model.fit(tf_dataset)

    def predictParty(self, sentence: List[str]):
        # Process the input
        encodeInput = self.tokenizer(sentence, truncation=True, padding=True)
        inputDataset = tf.data.Dataset.from_tensor_slices(dict(encodeInput))

        predictions = self.model.predict(inputDataset)
        predictedParty = tf.argmax(predictions.logits, axis=1)
        print(predictedParty)
