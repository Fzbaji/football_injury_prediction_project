Collecte des Données
=====================

Dans cette section, nous décrivons le processus de collecte et de création des données utilisées dans le projet. Deux types de données ont été considérés : les données tabulaires et les images. Tandis que les données tabulaires ont été générées synthétiquement pour répondre aux besoins spécifiques du projet, les images ont été collectées manuellement pour les besoins de classification des blessures.

Données tabulaires
------------------

Contexte
--------

La disponibilité de jeux de données fiables est cruciale pour l’entraînement des modèles d’apprentissage automatique. Cependant, en travaillant avec des datasets réels, plusieurs biais et limitations ont été constatés :

- Problèmes de qualité des données : certaines variables étaient mal renseignées ou manquantes.
- Biais dans la population : les datasets réels présentaient un manque de représentativité de certaines tranches d’âge et niveaux d’entraînement.
- Coût et accessibilité : certaines données provenant de bases de recherche sportive n’étaient pas accessibles ou étaient protégées par des restrictions légales.

Pour pallier ces limitations, nous avons opté pour une approche de **génération de données synthétiques** basée sur des études et analyses fiables.

Génération des Données Synthétiques
-----------------------------------

Les données synthétiques ont été créées à partir de distributions statistiques et d'hypothèses dérivées d'études scientifiques et de données existantes. Ces données permettent d'étudier les facteurs prédictifs des blessures musculaires, notamment dans le contexte du football, où ce type de blessure est l’un des plus critiques. 

**Variable Target**

La variable cible (*target*) est **binaire** et contient deux valeurs possibles :

- **1 :** Le joueur présente un risque de blessure musculaire.
- **0 :** Le joueur ne présente aucun risque de blessure musculaire.

Cette variable cible a été définie sur la base des conclusions de plusieurs études et documents fiables, qui montrent que certains facteurs (tels que l’historique de blessures musculaires ou des charges d’entraînement excessives) augmentent considérablement le risque. En particulier, les blessures musculaires sont une des préoccupations majeures dans le domaine du football, en raison de leur fréquence et de leur impact sur les performances des joueurs.

**Features Utilisées**

Voici les **features** principales incluses dans le jeu de données et leur pertinence dans le contexte de l’étude sur les blessures musculaires :

1. **Âge** :
   - Variable continue exprimée en années.
   - Impact potentiel sur le risque de blessure dû à la diminution de la récupération musculaire avec l'âge.

2. **Fatigue (%)** :
   - Mesure de la fatigue subjective exprimée en pourcentage (de 0 % à 100 %).
   - Déterminée à partir d’études sur la performance physique et la charge cumulée.

3. **Minutes jouées** :
   - Total des minutes jouées lors des compétitions.
   - Indicateur direct de la charge d'activité physique.

4. **Heures d’entraînement** :
   - Nombre d'heures d’entraînement.
   - Variable continue influençant les niveaux de fatigue et les risques musculaires.

5. **Historique des blessures musculaires** :
   - Variable binaire ou catégorielle indiquant des blessures musculaires récentes.
   - Facteur prédictif clé dans les études sur la prévention des blessures, les joueurs ayant déjà souffert de telles blessures étant souvent plus vulnérables.

6. **Contact physique (oui/non)** :
   - Variable binaire indiquant si l’athlète est impliqué dans des activités à contact physique élevé.
   - Ce facteur est essentiel, car les sports impliquant des contacts physiques augmentent les risques de blessure musculaire.

En résumé, ces variables ont été sélectionnées en fonction de leur pertinence scientifique et de leur contribution à prédire la variable cible. Ce choix garantit une meilleure compréhension et une analyse fiable des facteurs influençant les blessures musculaires.


..
    Visualisation des Données Générées   (comment la visualiser)
    ----------------------------------

    Pour valider la cohérence des données, plusieurs techniques de visualisation ont été utilisées:

    - Distribution des âges et des heures d’entraînement : **seaborn** a permis de générer des histogrammes pour vérifier que les valeurs suivent les attentes définies.
    - Matrice de corrélation : pour s’assurer de la pertinence des relations entre les variables générées.
    - Validation croisée : en utilisant des échantillons de validation synthétique.

Avantages des Données Synthétiques
-----------------------------------

L’utilisation des données synthétiques présente plusieurs avantages :

- **Absence de contraintes légales** : pas de restrictions associées aux données réelles.
- **Contrôle total sur les variables** : génération des données en fonction des hypothèses et des besoins spécifiques du projet.
- **Équilibrage des classes** : réduction des biais en équilibrant les observations avec et sans blessures.

---

Collecte des Données
=====================

Dans cette section, nous décrivons le processus de collecte et de création des données utilisées dans le projet. Deux types de données ont été considérés : les données tabulaires et les images. Tandis que les données tabulaires ont été générées synthétiquement pour répondre aux besoins spécifiques du projet, les images ont été collectées manuellement pour les besoins de classification des blessures.

---

Données d’Images pour la Classification des Blessures
------------------------------------------------------

### Contexte

Pour la deuxième partie du projet, visant la classification des types de blessures à partir d’images, un jeu de données d’images a été collecté manuellement. La collecte a été un défi majeur en raison des limitations suivantes :

1. **Disponibilité des données** :
   - Peu de bases de données publiques sur les blessures sportives sont disponibles, particulièrement celles avec des images classifiées selon des types de blessures spécifiques.
   - Les images disponibles étaient souvent dispersées sur des sources limitées et non standardisées.

2. **Manque de diversité et d’équilibre** :
   - Certaines classes de blessures étaient sur-représentées, comme les lésions courantes, tandis que d'autres (ex. : *ACL*) étaient sous-représentées.

### Processus de Collecte

Un effort manuel a été fait pour rassembler un ensemble d'images correspondant aux quatre classes identifiées dans le projet :

- **Classes de blessures** :
  0. Lésion
  1. Hamstring
  2. Entorse_Cheville
  3. ACL

Les images ont été collectées via des recherches manuelles et des bases médicales disponibles en ligne, tout en veillant à préserver la conformité légale et éthique.

### Augmentation des Données pour Équilibrer les Classes

Étant donné le volume limité d'images disponibles, en particulier pour les classes sous-représentées, des techniques d'augmentation d’images ont été employées :

- **Méthodes utilisées** :
  - Rotation
  - Réflexion horizontale ou verticale
  - Ajustement de la luminosité ou du contraste
  - Décalages et transformations aléatoires
  
- **Impact** :
  Ces augmentations ont aidé à créer un jeu de données plus équilibré, bien que le volume de données demeure relativement faible pour un problème de classification d’images complexe.

### Limitations

Malgré les efforts, certaines limitations subsistent :

1. Le jeu de données global reste modeste en taille, ce qui peut limiter les performances des modèles CNN.
2. La collecte manuelle entraîne des variations potentielles dans la qualité des images.


Conclusion
----------

La collecte manuelle d’images et l'augmentation des données ont permis de constituer une base suffisamment représentative pour entraîner un modèle de classification de blessures. Bien que des défis demeurent liés à la qualité et à la taille des données, cette étape constitue une avancée majeure dans l’analyse des images médicales de blessures sportives.

Pour continuer, voir la section suivante :source:`preprocessing`.

