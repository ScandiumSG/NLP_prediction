from load import DataFetcher
from processData import DataProcessor
from PredictionModel import PredictionClassifier
from in_progress.CNNModel import CNNModel
from in_progress.TransformerModel import BertModel
from testTransformerModelPrediction import testPrediction

fetcher = DataFetcher()
processor = DataProcessor()

dataset = fetcher.getSmallDataset()
dictionaryList = processor.jsonToDictionaryList(dataset)

partyList = processor.replacePersonWithParty(dictionaryList)

#model = PredictionClassifier(model="Forest", vectorizer="tfidf")
model = BertModel()
print("Starting training")
sentence = "Siden regjeringens pressekonferanse i slutten av september i fjor har vi sett en brå oppbremsing av en rekke investeringer i Norge. Vi har opplevd at uroen ute er supplert av en uro hjemme for fremtidsutsiktene på grunn av regjeringens skattepolitikk. Vi opplever betydelig uro i mange lokalsamfunn – uro for hvor jobbene deres går, uro for fremtidsutviklingen for flere av Norges viktigste næringer – og vi opplever at både børssjefen og andre analytikere sier at vi har innført politisk risiko ved investeringer som påvirker utlendingers vilje til å investere i Norge. Vi er altså dårligere rustet for å få til de store omstillingene vi trenger, og har mindre mulighet for å hente kapital internasjonalt hvis dette bildet fortsetter. Vi har sett stor stopp i kraftinvesteringene, vi har sett stor stopp i investeringer knyttet til laksenæringer. Jeg har derfor god forståelse for hvorfor statsministeren og finansministeren ved starten av sin budsjettkonferanse følte behov for å forsøke å berolige landet om skattepolitikken til regjeringen fremover, for da var budskapet fra statsministeren ingen generelt økte skatter i Norge fremover i resten av denne perioden, og fra finansministeren at det samlede skattetrykket ikke skal opp. Hvis dette skal virke beroligende, er det kanskje viktig for folk og resten av oss å vite: Hva definerer man da som det samlede skattetrykket? Fra hvilket nivå er det regjeringen tenker at man ikke skal øke skattene videre? Er det fra årets budsjett, eller er det fra årets budsjett minus de midlertidige skattene? "
sentence2 = "Når det gjelder høyprisbidraget, la meg minne om at det gjelder ekstraordinært høye inntekter til kraftselskaper som vi har bidratt til å fordele for å kunne betale strømstøtte og løse andre velferdsoppgaver. Det er varslet at det kommer til å bli faset ut innen utgangen av 2024. Den midlertidige arbeidsgiveravgiften vil også være midlertidig, men vi vil se på muligheter for at også den gradvis kan trappes ned. Det som er viktig for selskaper som vil investere, er at vi har en selskapsskatt på 22 pst. Den er trappet ned gjennom årene hvor Erna Solberg var statsminister. Vi har sagt at den skal ligge fast. Utover det – for inntekter og formue – er det om lag det systemet vi nå har. Vi får se i det endelige budsjettet hvordan dette kommer til å bli utformet, men det var viktig å gi et signal når regjeringen arbeider med et statsbudsjett og planlegger for det. Skal vi ha virkelig gode kår for å investere i Norge med store muligheter, er det signalet sendt, og da har jeg tro på at vi også kan videreføre et høyt aktivitetsnivå, lav ledighet og flere i jobb også i 2023."

model.trainModel2(dataset)