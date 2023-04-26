from typing import Dict, List
from load import DataFetcher
from processData import DataProcessor

class testPrediction():
    corpus = []

    def __init__(self):
        # By Erna Solberg (H), 22.03.2023
        sentence = "Siden regjeringens pressekonferanse i slutten av september i fjor har vi sett en brå oppbremsing av en rekke investeringer i Norge. Vi har opplevd at uroen ute er supplert av en uro hjemme for fremtidsutsiktene på grunn av regjeringens skattepolitikk. Vi opplever betydelig uro i mange lokalsamfunn – uro for hvor jobbene deres går, uro for fremtidsutviklingen for flere av Norges viktigste næringer – og vi opplever at både børssjefen og andre analytikere sier at vi har innført politisk risiko ved investeringer som påvirker utlendingers vilje til å investere i Norge. Vi er altså dårligere rustet for å få til de store omstillingene vi trenger, og har mindre mulighet for å hente kapital internasjonalt hvis dette bildet fortsetter. Vi har sett stor stopp i kraftinvesteringene, vi har sett stor stopp i investeringer knyttet til laksenæringer. Jeg har derfor god forståelse for hvorfor statsministeren og finansministeren ved starten av sin budsjettkonferanse følte behov for å forsøke å berolige landet om skattepolitikken til regjeringen fremover, for da var budskapet fra statsministeren ingen generelt økte skatter i Norge fremover i resten av denne perioden, og fra finansministeren at det samlede skattetrykket ikke skal opp. Hvis dette skal virke beroligende, er det kanskje viktig for folk og resten av oss å vite: Hva definerer man da som det samlede skattetrykket? Fra hvilket nivå er det regjeringen tenker at man ikke skal øke skattene videre? Er det fra årets budsjett, eller er det fra årets budsjett minus de midlertidige skattene?"
        self.corpus.append({"Speaker": "Erna Solberg", "Party": "H", "Text": sentence})

        # By Jonas Gahr Støre (AP), 22.03.2023
        sentence2 = "Når det gjelder høyprisbidraget, la meg minne om at det gjelder ekstraordinært høye inntekter til kraftselskaper som vi har bidratt til å fordele for å kunne betale strømstøtte og løse andre velferdsoppgaver. Det er varslet at det kommer til å bli faset ut innen utgangen av 2024. Den midlertidige arbeidsgiveravgiften vil også være midlertidig, men vi vil se på muligheter for at også den gradvis kan trappes ned. Det som er viktig for selskaper som vil investere, er at vi har en selskapsskatt på 22 pst. Den er trappet ned gjennom årene hvor Erna Solberg var statsminister. Vi har sagt at den skal ligge fast. Utover det – for inntekter og formue – er det om lag det systemet vi nå har. Vi får se i det endelige budsjettet hvordan dette kommer til å bli utformet, men det var viktig å gi et signal når regjeringen arbeider med et statsbudsjett og planlegger for det. Skal vi ha virkelig gode kår for å investere i Norge med store muligheter, er det signalet sendt, og da har jeg tro på at vi også kan videreføre et høyt aktivitetsnivå, lav ledighet og flere i jobb også i 2023."
        self.corpus.append({"Speaker": "Jonas Gahr Støre", "Party": "AP", "Text": sentence2})

        # By Kirsti Bergstø (SV), 22.03.2023
        sentence3 = "Verden trenger en enorm omstilling for å nå klimamålene, og vi trenger en stor omstilling i eget samfunn for industri og arbeidsplasser. Verdens klimaforskere sier at den siste rapporten er en siste advarsel. I USA ser vi at Biden lanserer «inflation reduction act», som på kort sikt har skapt lovnader om 100 000 arbeidsplasser. Her hjemme er det samlede budskapet fra LO og NHO at vi mangler om lag 115 000 arbeidsfolk for å sikre det grønne skiftet. Vi vet at EU akkurat nå også jobber med en stor grønn industripakke, som vil møte USAs grønne kjempe. Det er en pakke som Norge enn så lenge velger å stå på utsiden av, ikke fordi den ikke er god nok, men fordi regjeringen ikke er enig med EU i at oljeutvinning i Arktis er dårlig. Verden legger nå om tempoet for å bygge ny, grønn industri og for å nå klimamålene. Dommen fra FNs klimapanel er at alle oljefeltene som allerede er i produksjon, vil føre til at vi ikke når klimamålene og 1,5-gradersmålet. Ved ikke å gire opp akselererer både klimakrisen og faren for industridød. Da bør vi etter SVs syn si ja til EUs industriplan og nei til mer oljeleting i Arktis. Derfor lurer jeg på hvorfor olje og gass i Arktis virker å være viktigere for statsministeren enn en industriell omstilling." 
        self.corpus.append({"Speaker": "Kirsti Bergstø", "Party": "SV", "Text": sentence3})

        # By Guri Melby (V), 22.03.2023
        sentence4 = "Det er veldig bra at det nå ser ut til at Norge blir enig med EU om en grønn omstilling, men for at vi skal lykkes med dette, er jeg helt sikker på at vi trenger å implementere mer av EUs klimapolitikk i Norge. Det er åpenbart at det er en uenighet mellom de to regjeringspartiene i synet på EUs ren energi-pakke. Mitt konkrete spørsmål til statsministeren er: Kan han garantere at den regjeringen han leder, vil gjennomføre EUs ren energi-pakke i Norge?"
        self.corpus.append({"Speaker": "Guri Melby", "Party": "V", "Text": sentence4})

        # By Bjørnar Moxnes (R), 22.03.2023
        sentence5 = "Det er jo helt glimrende hvis statsminister Støre nå har rett i at ACER ikke står i veien for at vi kan sette en innenlands makspris på strøm og skille mellom prisingen innenlands og det som er prisen på eksport. Det er så fall glimrende og interessant for det norske folk, skulle jeg tro. Et annet ufravikelig krav var at myndighetene i Norge skal ha selvstendig kontroll over alle avgjørelser med betydning for energisikkerheten. Da regjeringen skulle lage en styringsmekanisme for vannmagasinene, snakket de hele høsten om å begrense eksporten. Det utredet RME, reguleringsmyndigheten for energi, og viste til en hjemmel i elmarkedsdirektivet til EU som åpnet for kortsiktige krisetiltak, men ikke for langvarige tiltak for å ivareta energisikkerheten. Resultatet var ikke det regjeringen hadde gitt inntrykk av, men det var en begrensning på produksjon, og bare på eksport i helt ekstreme tilfeller, med det resultat at den ordningen som kommer, vil presse strømprisene opp. Er statsminister Støre enig i at vi mangler selvstendig kontroll om vi ikke kan begrense eksport som et langvarig tiltak for energisikkerhet?"
        self.corpus.append({"Speaker": "Bjørnar Moxnes", "Party": "R", "Text": sentence5})

    def checkSentences(self, model, model_label):
        for dict in self.corpus:
            print(f"{model_label} predicted :", model.predictParty(dict['Text']), f"\tShould be: {dict['Party']} - {dict['Speaker']}")

    def checkSentenceWithProbability(self, model, sentence):
            print("Probability for sentence")
            print(model.predict_proba(sentence))