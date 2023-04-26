from load import DataFetcher
from processData import DataProcessor
from PredictionModel import PredictionClassifier
from testModelPrediction import testPrediction

fetcher = DataFetcher()
processor = DataProcessor()

dataset = fetcher.getFullDataset()
dictionaryList = processor.jsonToDictionaryList(dataset)

models = []

print("(Count-NB)")
speaker_model = PredictionClassifier(model="nb", vectorizer="count")
speaker_model.trainModel(dictionaryList)
models.append(("(Count-NB)", speaker_model))


print("(Count-NB)")
partyList = processor.replacePersonWithParty(dictionaryList)
party_model = PredictionClassifier(model="nb", vectorizer="count")
party_model.trainModel(partyList)
party_model.generateConfusionMatrix("Count-NB")
models.append(("(Count-NB)", party_model))

print("(TF_IDF-NB)")
tfidfModel = PredictionClassifier(model="nb", vectorizer="tfidf")
tfidfModel.trainModel(partyList)
tfidfModel.generateConfusionMatrix("TF_IDF-NB")
models.append(("(TF_IDF-NB)", tfidfModel))

print("(Count-Compliment)")
complimentNB2 = PredictionClassifier(model="compliment", vectorizer="count")
complimentNB2.trainModel(partyList)
complimentNB2.generateConfusionMatrix("Count-Compliment")
models.append(("(Count-Compliment)", complimentNB2))


print("(TF_IDF-Compliment)")
complimentNB = PredictionClassifier(model="compliment", vectorizer="tfidf")
complimentNB.trainModel(partyList)
complimentNB.generateConfusionMatrix("TF_IDF-Compliment")
models.append(("(TF_IDF-Compliment)", complimentNB))


enableForest = False
if enableForest:
    forestCount = 30
    # Disable forest if you dont have 10min to generate...
    print("(Count-Forest)")
    forest_count = PredictionClassifier(model="forest", vectorizer="count", forestCount=forestCount)
    forest_count.trainModel(partyList)
    forest_count.generateConfusionMatrix("Count-forest")
    models.append(("(Count-Forest)", forest_count))

    print("(TF_IDF-Forest)")
    forest_tfidf = PredictionClassifier(model="forest", vectorizer="tfidf", forestCount=forestCount)
    forest_tfidf.trainModel(partyList)
    forest_tfidf.generateConfusionMatrix("TF_IDF-forest")
    models.append(("(TF_IDF-Forest)", forest_tfidf))


sentence = {"Speaker": "Erna Solberg", "Party": "H", "Text": "Siden regjeringens pressekonferanse i slutten av september i fjor har vi sett en brå oppbremsing av en rekke investeringer i Norge. Vi har opplevd at uroen ute er supplert av en uro hjemme for fremtidsutsiktene på grunn av regjeringens skattepolitikk. Vi opplever betydelig uro i mange lokalsamfunn – uro for hvor jobbene deres går, uro for fremtidsutviklingen for flere av Norges viktigste næringer – og vi opplever at både børssjefen og andre analytikere sier at vi har innført politisk risiko ved investeringer som påvirker utlendingers vilje til å investere i Norge. Vi er altså dårligere rustet for å få til de store omstillingene vi trenger, og har mindre mulighet for å hente kapital internasjonalt hvis dette bildet fortsetter. Vi har sett stor stopp i kraftinvesteringene, vi har sett stor stopp i investeringer knyttet til laksenæringer. Jeg har derfor god forståelse for hvorfor statsministeren og finansministeren ved starten av sin budsjettkonferanse følte behov for å forsøke å berolige landet om skattepolitikken til regjeringen fremover, for da var budskapet fra statsministeren ingen generelt økte skatter i Norge fremover i resten av denne perioden, og fra finansministeren at det samlede skattetrykket ikke skal opp. Hvis dette skal virke beroligende, er det kanskje viktig for folk og resten av oss å vite: Hva definerer man da som det samlede skattetrykket? Fra hvilket nivå er det regjeringen tenker at man ikke skal øke skattene videre? Er det fra årets budsjett, eller er det fra årets budsjett minus de midlertidige skattene? "}
sentence2 = {"Speaker": "Jonas Gahr Støre", "Party": "AP", "Text": "Når det gjelder høyprisbidraget, la meg minne om at det gjelder ekstraordinært høye inntekter til kraftselskaper som vi har bidratt til å fordele for å kunne betale strømstøtte og løse andre velferdsoppgaver. Det er varslet at det kommer til å bli faset ut innen utgangen av 2024. Den midlertidige arbeidsgiveravgiften vil også være midlertidig, men vi vil se på muligheter for at også den gradvis kan trappes ned. Det som er viktig for selskaper som vil investere, er at vi har en selskapsskatt på 22 pst. Den er trappet ned gjennom årene hvor Erna Solberg var statsminister. Vi har sagt at den skal ligge fast. Utover det – for inntekter og formue – er det om lag det systemet vi nå har. Vi får se i det endelige budsjettet hvordan dette kommer til å bli utformet, men det var viktig å gi et signal når regjeringen arbeider med et statsbudsjett og planlegger for det. Skal vi ha virkelig gode kår for å investere i Norge med store muligheter, er det signalet sendt, og da har jeg tro på at vi også kan videreføre et høyt aktivitetsnivå, lav ledighet og flere i jobb også i 2023."}
evalSentence = sentence

#for (Label, model) in models:
#    print(f"Prediction {Label}: ", model.predictParty(evalSentence["Text"]), f"\nCorrect value: {evalSentence['Speaker']} - {evalSentence['Party']}")

test = testPrediction()
for (Label, model) in models:
    test.checkSentences(model=model, model_label=Label)
    
    method = getattr(model, 'predict_proba', None)
    if (method):
        sentence4 = "Det er veldig bra at det nå ser ut til at Norge blir enig med EU om en grønn omstilling, men for at vi skal lykkes med dette, er jeg helt sikker på at vi trenger å implementere mer av EUs klimapolitikk i Norge. Det er åpenbart at det er en uenighet mellom de to regjeringspartiene i synet på EUs ren energi-pakke. Mitt konkrete spørsmål til statsministeren er: Kan han garantere at den regjeringen han leder, vil gjennomføre EUs ren energi-pakke i Norge?"
        test.checkSentenceWithProbability(model=model, model_label=Label, sentence=sentence4)
    print("")

