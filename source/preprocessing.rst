..
   Le prétraitement est une étape cruciale dans tout projet d’apprentissage automatique.
   Voici les principales étapes :

   1. **Nettoyage des données** :
      - Gestion des valeurs manquantes : imputations par la moyenne ou la médiane.
      - Suppression des doublons.

   2. **Encodage des variables catégoriques** :
      - Utilisation de l'encodage one-hot pour les positions des joueurs.

   3. **Normalisation et standardisation** :
      - StandardScaler de Scikit-learn a été utilisé pour uniformiser les valeurs.
   
Prétraitement des Données
=========================

Dans cette section, nous décrivons le processus de prétraitement appliqué aux données synthétiques et aux images pour garantir leur qualité et leur adaptabilité aux modèles d'apprentissage automatique. Ce processus inclut la gestion des valeurs manquantes, l’encodage des variables catégoriques, la séparation des ensembles d'entraînement et de test, le Redimensionnement, ainsi que l’équilibrage des classes.

Donnés tabulaires
-----------------

**Détection et Gestion des Valeurs Manquantes**


Les valeurs manquantes ont été détectées et traitées comme suit :

.. code-block:: python

    # Vérifier les valeurs manquantes
    data.isnull().sum()

    # Supprimer les lignes avec des valeurs manquantes
    data.dropna(inplace=True)

L'objectif était d'assurer la qualité des données en éliminant les observations incomplètes, minimisant ainsi l’impact sur les performances du modèle.

**Encodage des Variables Catégoriques**

Pour rendre les données exploitables par les algorithmes d’apprentissage automatique, les variables catégoriques ont été encodées en variables numériques :

.. code-block:: python

    # Encodage des variables catégoriques
    data = pd.get_dummies(data, drop_first=True)


**Séparation des Features et du Label**

Les données ont ensuite été séparées en caractéristiques (*features*) et en labels (*target*) :

.. code-block:: python

    # Séparer les features (X) et le label (y)
    X = data.drop(columns=['Blessure musculaire'])
    y = data['Blessure musculaire']

**Séparation des Ensembles d'Entraînement et de Test**


Un ensemble d’entraînement (80 % des données) et un ensemble de test (20 % des données) ont été créés pour évaluer les performances des modèles :

.. code-block:: python

    # Séparer les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

**Équilibrage des Classes**

Étant donné un déséquilibre des classes (une sous-représentation des blessures musculaires), une méthode de sur-échantillonnage a été appliquée à l'ensemble d'entraînement pour équilibrer les classes :

.. code-block:: python

    # Fusionner X_train et y_train
    train_data = pd.concat([X_train, y_train], axis=1)

    # Séparer les classes majoritaires et minoritaires
    majority_class = train_data[train_data['Blessure musculaire'] == 0]
    minority_class = train_data[train_data['Blessure musculaire'] == 1]

    # Sur-échantillonner la classe minoritaire
    minority_upsampled = resample(minority_class,
                                  replace=True,
                                  n_samples=len(majority_class),
                                  random_state=42)

    # Combiner les classes
    train_data_balanced = pd.concat([majority_class, minority_upsampled])

    # Séparer les features et le label
    X_train_balanced = train_data_balanced.drop(columns=['Blessure musculaire'])
    y_train_balanced = train_data_balanced['Blessure musculaire']

Grâce à cette technique, les données sont désormais équilibrées, ce qui améliore les performances des modèles d'apprentissage en réduisant le biais en faveur de la classe majoritaire.

---

Données d'images
----------------

**Organisation des Données**

Les données d'images collectées ont été classées manuellement en 4 catégories correspondant aux types de blessures :
    - Lésion
    - Hamstring
    - Entorse de cheville
    - Rupture du ligament croisé (ACL)

Les données d’images ont été divisées en trois ensembles pour garantir une évaluation rigoureuse des modèles :
    - **Ensemble d’entraînement** (70 % des images) : utilisé pour ajuster les paramètres du modèle.
    - **Ensemble de validation** (20 % des images) : utilisé pour affiner les hyperparamètres.
    - **Ensemble de test** (10 % des images) : réservé pour une évaluation finale après l’entraînement.



**Prétraitement et Génération des Données**

Afin de préparer les données pour un réseau de neurones convolutif (CNN), les images ont été transformées comme suit :
    1. **Redimensionnement** : Toutes les images ont été redimensionnées à une taille uniforme de 150x150 pixels.
    2. **Normalisation** : Les valeurs des pixels ont été mises à l’échelle dans l’intervalle [0, 1].

Le chargement et le prétraitement des images ont été réalisés à l’aide de la classe `ImageDataGenerator`, qui applique ces transformations automatiquement pendant l’entraînement.

Exemple de configuration pour l’entraînement et la validation :

.. code-block:: python

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Préparation des générateurs
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

À l'étape suivante, les données prétraitées seront utilisées pour entraîner les modèles. Reportez-vous à la section suivante : :doc:`model_training`.

