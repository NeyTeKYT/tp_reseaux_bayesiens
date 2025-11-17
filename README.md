# 🧠 Gestion de l'incertitude avec les réseaux bayésiens

Bienvenue sur le dépôt de mon **TP de troisième année de licence informatique**, réalisé dans le cadre de l'UE *Initiation à l'IA* à l'Université d'*Avignon*.  

Ce TP porte sur la **modélisation et l'inférence dans des réseaux bayésiens**, appliquées à la génétique et à l'hérédité.  

---

## 📋 Prise en main de TabularCPD

La première étape du TP consistait à reproduire plusieurs tableaux de probabilités, notamment : 
- la **probabilité du nombre de versions mutées du gène chez une personne** (Figure 1.a), via la fonction `get_probs_gene_ancestor(varName)`.
- la **probabilité de l'apparation d'un trait en fonction du nombre de versions mutées du gène** (Figure 1.b), via la fonction `get_probs_trait(varName,evidenceName)`.  

Cette première étape m'a vraiment permis de **comprendre en profondeur le fonctionnement des TabularCPD**, fondamental pour la suite du TP !  

---

## 👶 Modélisation de l'hérédité

La deuxième étape a consisté à compléter la fonction `get_probs_heredity1(geneParent)` afin de déterminer la **probabilité conditionnelle P(G enfant | G père, G mère)**.  

Autrement dit : calculer la probabilité qu'un chromosome hérité d'un parent (père ou mère) porte exactement **une version mutée**, en connaissant le nombre de mutations présentes chez ce parent.  

Cette implémentation a permis de finaliser la fonction `get_probs_gene(varNameChild,evidenceNameFather,evidenceNameMother)`, qui calcule toutes les probabilités possibles pour l'enfant selon les combinaisons de gènes parentaux.  

---

## 👪 Étude de la famille n°1 

Une fois les fonctions de base implémentés, j'ai modélisé un premier cas d'une famille (Figure 3) en utilisant ces dernières.
Le modèle obtenu a été vérifié à la fois **graphiquement** avec la génération d'une image `family1.png`, et grâce à la fonction `check_model()` afin de garantir la cohérence des probabilités.  

Deux méthodes d'inférence ont été utilisées : 
- **Inférence exacte :** estimation du nombre de gènes mutés chez Paul à partir de la présence ou non du trait dans sa famille, puis en intégrant les connaissances sur les mutations des autres individus.
- **Inférence approchée par échantillonnage :** simulation d'un très grand nombre de cas compatibles avec le problème, puis estimation des probabilités.  

En comparant les résultats, on constate que les différences entre inférence exacte et inférence approchée sont minimes; néanmoins, lorsque l'on recherche une valeur exacte, l'inférence approchée reste insuffisante, même si elle fournit une estimation très fiable.

---

## 👑 Étude de la famille n°2 (famille royale)

La dernière étape du TP consistait à appliquer les mêmes méthodes à une **famille plus complexe : la famille royale**.  

Cette fois, l'objectif était de calculer les probabilités concernant *George* et *Liliet*, avec un arbre généalogique plus large et plus riche en dépendances.

---

## ⚙️ Installation

Avant d'exécuter le programme `genetics.py`, lancez les commandes suivantes issues du fichier `CONFIG.txt : 
- `conda activate init-IA`
- `conda install python=3.11`
- `conda install -c conda-forge pgmpy`
- `conda install conda-forge::opt_einsum conda-forge::pyro-ppl`
- `conda install conda-forge::pygraphviz conda-forge::xgboost`

Une fois l'environnement configuré, exécutez le programme avec : `python3 genetics.py`.
