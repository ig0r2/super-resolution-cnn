# Super Resolution projekat

Ovaj projekat ima za cilj implementaciju i upoređivanje različitih konvolucionih modela za povećanje rezolucije
slika (Single Image Super-Resolution)

## Uvod

Klasično povećanje rezolucije zasniva se na matematičkoj interpolaciji. Ove metode služe kao bazna linija za poređenje
sa neuronskim modelima.

- Nearest Neighbor - Najjednostavnija metoda interpolacije, svaki novi piksel dobija vrednost najbližeg postojećeg
  piksela.
- Bilinear - Koristi linearnu kombinaciju 4 susedna piksela.
- Bicubic - Koristi 4×4 okolinu (16 piksela) i kubnu interpolaciju.
- Lanczos - Koristi sinc funkciju.

*Klasični metodi se mogu koristiti preko klase `RegularModel` iz `models/regular_models.py`*

Za razliku od interpolacije, neuronske mreže pokušavaju da nauče mapiranje između slike niske rezolucije (LR) i slike
visoke rezolucije (HR).

- Prvi uspešan CNN model za super rezoluciju bio je SRCNN. Koristio je plitku konvolucionu mrežu koja je obrađivala samo
  Y kanal (luminance) nad ulazom koji je već uvećana slika pomoću Bicubic interpolacije.
  https://arxiv.org/pdf/1501.00092

- Kasnije je isproban model SRResNet koji je baziran na ResNet arhitekturi i koji uči direktno upsamplovanje cele slike.
  https://arxiv.org/pdf/1609.04802

## Implementirani modeli:

- ### **EDSR** (Enhanced Deep Super-Resolution)

https://arxiv.org/pdf/1707.02921

Poboljšana verzija SRResNet arhitekture. Koristi rezidualne blokove ali je uklonjena Batch normalizacija, što je dovelo
do smanjenja memorije i povećanja brzine.
*Detaljniji opis je u komentarima klase. `models/edsr/model.py`*

- ### **IMDN** (Information Multi-Distillation Network)

https://arxiv.org/pdf/1909.11856v1

Laganija arhitektura sa ciljem da bude efikasnija od prethodnih modela.
Koristi destilaciju feature-a unutar svakog IMD bloka (IMDB). Na izlazu iz blokova se koristi i CCA (Contrast-aware
Channel Attention).
*Detaljniji opis je u komentarima klase. `models/imdn/model.py`*

- ### **RFDN** (Residual Feature Distillation Network)

https://arxiv.org/pdf/2009.11551

RFDN je direktno unapredjenje IMDN mreze.
IMDN mreza destiluje feature pomocu fiksne podele izlaza, dok RFDN dozvoljava mrezi da nauci sama koje feature da
destiluje. Takodje, umesto channel attention RFDN koristi ESA (Enhanced Spatial Attention) koji uci
prostorni attention medju piskelima. `models/rfdn/model.py`

- ### Moja verzija (CustomERN)

Koristi blokove inspirisane Inverted Residualnim blokovima.
Blokovi rade ekspanziju kanala pointwise konvolucijom a zatim skupljaju kanale 3x3 konvolucijom.
Mreža ima jedan konvolucioni sloj pre i posle blokova. Nema nikakav attention.
*Detaljniji opis je u komentarima klase. `models/custom/ern.py`*

## Struktura projekta

### Folderi

`checkpoints/`

- Sacuvani trenirani modeli (.pth)

`config/`

- Konfiguracioni YAML fajlovi koji se koriste za treniranje

`data/`

- Podaci za treniranje, test i validaciju
- Skalirane i originalne verzije slika (LR/HR parovi)
- Podaci se automatski preuzimaju pri prvom korišćenju odgovarajućeg dataseta

`datasets/`

- `ImageDatasetTrain` i `ImageDatasetTest` klase (`torch.utils.data.Dataset`)
- Funkcije za kreiranje klase, preuzimanje i transformaciju datasetova

`inference/`

- Folder za slike za inference i comparison skripte

`logs/`

- Logovi treniranja i evaluacije
- Grafici treniranja

`models/`

- Implementacije modela
- model_registy

`results/`

- Rezultati evaluacija u CSV formatu

`utils/`

- Pomoćne klase i funkcije

### Skripte (root)

### `training_script.py`

- Glavna skripta za treniranje modela
- Konfiguracija preko YAML fajlova iz `config/`
- Treniranje preko klase `Trainer` (`utils/trainer.py`)

### `evaluate_script.py`

- Evaluacija modela na test skupu
- Izračunavanje LPIPS, SSIM, PSNR metrika preko klase `Evaluator` iz `utils/evaluator.py`
- Merenje inference brzine preko klase `EvaluatorPerf` iz `utils/evaluator_perf.py`

### `inference_script.py`

- Batch inference nad svim slikama iz `inference/input/`
- Čuvanje uvećanih slika u `inference/output/`

### `comparison_script.py`

- Datu sliku uvećava koristeći razlićite metode i modele
- Pravi grid od delova uvećanih slika kako bi se videla vizuelna razlika između različitih modela
- Rezultat u `inference/comparison/`

## Rezultati testiranja za 2x modele

Svi modeli su trenirani sa konfiguracijom:
`patch_size: 96;
batch_size: 16;
epochs: 800;
lr: 0.0001;
scheduler_step_epochs: 300;
validate_every: 50;`

Korišćene metrike su PSNR, SSIM i LPIPS. Koriscen je `torchmetrics` paket za implementaciju metrika, klase su u
`utils/metrics.py`.
Kod evaluacije, PSNR i SSIM se računaju nad izlazom koji je prvo pretvoren u int8 vrednosti piksela, kako bi se izmerila
metrika nad realnim vrednostima piksela. PSNR i SSIM se računaju na svim kanalima, ne samo na Y kanalu (što se koristi u
naučnim radovima).

Tokom treniranja svi modeli su imali stabilan rast SSIM, i uglavnom su nakon 500 epoha imali vrlo mali rast.

#### FP32 vs FP16:

Testiranjem originalnih modela sa preciznošću FP32 u odnosu na model pretvorene u half preciznost (FP16), dobijamo da
nema značajne razlike u preciznosti modela ali da ima poveće razlike u brzini i memoriji. Maksimalna razlika za SSIM je
0.0001, a za PSNR je 0.0033, dok je brzina u proseku veća 2.2 puta i memorija manja 2.1 puta.
Ovo pokazuje da je FP16 vrlo pogodna opcija za inferencu bez gubitka kvaliteta.

FP32 rezultati su u `results/results_2x_DIV2K.csv`, FP16 rezultati su u `results/results_2x_DIV2K_half.csv`

### Rezultati

Sledeći zaključci o arhitekturama i raznim konfiguracijama su doneseni iz `results/results_2x_DIV2K_half.csv` rezultata:

U opsegu do približno 100 000 parametara modeli uglavnom ne prelaze SSIM vrednost od 0.92, ali prednost ovih
konfiguracija je izuzetno visoka brzina, kod 720p rezolucije moguće je dostići i preko 200 FPS.

U opsegu do oko 800 000 parametara pojavljuju se modeli koji postižu SSIM između 0.92 i 0.93. U ovom segmentu
većina konfiguracija može da dostigne minimum brzinu od 30 FPS za 720p, a neki mogu i vise od 60FPS. Ovaj opseg
predstavlja neki kompromis između brzine i preciznosti.

Iznad 800 000 parametara modeli dostižu SSIM vrednosti preko 0.93, ali rast kvaliteta postaje manji i ograničen (SSIM
maksimalno do 0.9359). U ovom opsegu je brzina vrlo mala.

Kada se porede arhitekture EDSR, RFDN i IMDN, ne može se izdvojiti apsolutni pobednik. Za svaku od ovih arhitektura
moguće je pronaći konfiguraciju koja u određenom SSIM i FPS opsegu daje vrlo slične rezultate. Drugim rečima,
performanse više zavise od konkretne konfiguracije nego od same arhitekture.

Kod RFDN i IMDN modela rezultati pokazuju da je često isplativije povećavati širinu (broj kanala u bloku) nego dubinu (
broj blokova). Povećanje broja blokova donosi relativno mali napredak u SSIM, ali značajno utiče na pad brzine (
verovatno zbog konkatenacije izlaza iz blokova). Sa druge strane, širi modeli sa manjim brojem blokova često postižu
sličan kvalitet uz znatno bolje performanse. Najbolje je imati do 2, eventualno 4 bloka.

Kod EDSR arhitekture poželjno je koristiti malo veći broj blokova, jer povećanje dubine ima bolji uticaj na kvalitet, a
pad brzine nije toliko drastičan kao kod RFDN i IMDN.

CustomERN arhitektura može dostići konkurentan kvalitet, ali to uglavnom postiže sa duplo većim brojem parametara u
odnosu na RFDN i IMDN konfiguracije sa sličnim SSIM vrednostima. Pored toga, brzina inferencije je u pravilu niža, što
ovu arhitekturu čini manje efikasnom.