# Latent-Space-manipulation---StyleGAN2
Seminarska naloga pri predmetu Seminar iz biometričnih sistemov (Univerza v Ljubljani, Fakulteta za elektrotehniko - Avtomatika in informatika (BMA))

---
## Abstract
Sinteza fotorealističnih obrazov je že vrsto let na področju biometrije precej aktualna tema. V zadnjem času
ta še posebej pridobiva na pozornosti saj so slike obrazov obravnavane kot osebni podatki in te ne smejo biti
prosto uporabljene v zbirkah brez lastnikovega dovoljenja. Ker so te osnovna oblika podatkov za učenje globokih nevronskih
mrež za razpoznavanje obrazov je ena od možnih rešitev predstavljena v tem članku. Generativne nasprotni
ške mreže (angl. Generative Adversarial Networks - GAN) so danes sposobne ustvariti skoraj nerazpoznavne replike
obrazov resničnih oseb. Z ustrezno manipulacijo lahko tem dodajamo ali odvzemamo določene lastnosti ter tako
dobimo variacije posamezne osebe. Raziskali smo možnost uporabe take nevronske mreže imenovane StyleGAN2 za
potrebe generiranja sintetične zbirke obrazov. Uspešnost poskusa smo ovrednotili z razpoznavalnikom obrazov Arc-
Face, ki uporablja ResNet-18 hrbtenico. Naučen je bil tako na umetno generiranih podatki kot na že uveljavljeni LFW
zbirki, ki vsebuje slike resničnih ljudi. Rezultati so pokazali, da tak način ni primeren za učenje razpoznavalnikov obrazov
in bi uporabljena metoda potrebovala dodatke nadgradnje, da bi bila ustrezna.

![present_img](https://github.com/TilenTinta/Sinteza-slik-obrazov-za-ucenje-modelov-za-razpoznavanje-obrazov/tree/main/Slike/teser.PNG)

---
## Zahtevana predpriprava
Za generiranje slik namenjenih iskanju latentnih smeri in uporabi manipuliranih vektorjev za granjenje slikovne baze je potrebno uporabiti prednaučen model
StyleGAN2 razvitega s strani Nvidia. Ta je dostopen na:
- https://github.com/NVlabs/stylegan2

Model lahko pri implementaciji povzroči velik t.i. "dependency hell" zato po mojih izkušnjah svetujem uporabo Docker kontejnerja, ki je ravno tako
na voljo na njihovem repozitoriju:
- (https://github.com/NVlabs/stylegan2/blob/master/Dockerfile)

V primeru treniranja svojega modela je potrebno prenesti še Labelled Faces in the Wild (LFW) Dataset:
- https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

---
## Opis kode
- faceGeneratorStyleGAN2.py - generiranje n števila slik iz naključnih latentnih vektorjev
- fastSort.py - prikaže vsako generirano sliko in jo mi razvrstimo v mapi z in brez neke lastnosti (očala, nasmeh, starost...)
- sortToSCV.py - na osnovi deljenih podatkov izdena .csv datoteko
- smerSVM.py - latentne vektorje iz razredov uvozi in izvede PCA analizo ter SVM da najde latentno smer
- featureTest.py - na osnovi poznane smeri predvideva ali nek latentni vektor ima lastnost ali ne (brez generiranja slike)
- generateWithFeature.py - na nekem latentnem vektorju generira zahtevano lastnost (potrebuje smeri) z predpisano alpho (ugotavljanje alphe)
- variationGenerator.py - iz osnovne identitete generira n variacij (smer, starost, očala, barva las...)
- datasetGenerator.py - generira n identitet in za vsako n variacij iz vseh podanih smeri >> ustvarjanje dataseta
- mereKakovosti.py - računanje raznolikosti podatkov
- aligner.py - poravnavanje obrazov in njihovo izrezovanje (LFW in naš dataset)
- ArcFace_ResNet.py - NN za razpoznavanje obrazov (train in test)