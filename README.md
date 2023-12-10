# What is this?

This repository contain the source code a natural language processing (NLP) project to use Naive Bayes classifier to predict the political affiliation of member of the Norwegian parliment.

The project is part of the project in the course [TDT4310](https://www.ntnu.no/studier/emner/TDT4310), Intelligent Text Analytics and Language Understanding, at NTNU.

# How to run

It it recommended to run the project in a virtual environment, but not a requirement. For how to set up the venv look [here](https://docs.python.org/3/library/venv.html).

Install the following:

```
pip install datasets
pip install -U scikit-learn
```

That should be it (for now), some files in the `in_progress` folder do require TensorFlow.

After the installation is complete simply navigate to the `src` folder, and run one of the scripts with the command:

```
python <script name>.py
```

The scripts do the following:

- script.py
  - Trains and evaluates the models. Prints the key metrics (accuracy and F1) to console.
- testPrediction.py
  - Trains, generates confusion matrix, and attempts to make a prediction on a few select, hard-coded, sentences. Results are printed to console.
- makePartyPlot.py
  - Makes a bar graph of each party and the number of documents associated with each party.

# How to predict?

At this point, the only way to make predictions is to add the text to the testModelPrediction.py class and run the testPrediction.py script.

