HINGREZ Alexandre - KAMACHE Mohand - LESTROHAN Céline -THEUNISSEN Mathilde

Bienvenue dans le projet Création d'une Talkbox numérique

Nous avions utilisé un autre git que celui que vous aviez créer.
Si vous souhaitez voir notre git afin d'évaluer l'avancement du projet, voici le lien :

https://github.com/Mojand/Projet_Son_Talkbox.git

Pour lancer le programme, effectuez la commande suivante :

Etape 1 :
	python3 etape1.py
Etape 2 :
	python3 traitement_audio.py

Des arguments peuvent être ajoutés afin de choisir :
	- le fichier .wav de parole
	- le fichier .wav de l'instrument (par défaut, il s'agit d'un bruit blanc)
	- la méthode de résolution des coefficients du filtre par LPC
	- l'ordre du filtre
	- l'ajout d'une pré-amplification (filtre dérivateur) si derive = True
	- l'affichage ou non des graphiques
	- le pourcentage de recouvrement entre les différentes trames
La commande suivante permet de connaître les arguments à passer au script python.
	python3 traitement_audio.py --help

Une fois le signal de l'instrument filtré, il est possible de l'écouter.
La piste audio se trouve dans le dossier audio sous le nom pyaudio_output.wav


