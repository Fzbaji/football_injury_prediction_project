Entraînement des Modèles
========================

Mdèle Tabulaire
----------------

Une étape clé de l’analyse a été de tester plusieurs algorithmes d’apprentissage supervisé pour prédire les risques de blessures musculaires. Les modèles suivants ont été entraînés en utilisant les données équilibrées après les étapes de prétraitement.

Plusieurs modèles populaires et performants pour des tâches de classification ont été testés :

1. **Régression Logistique** :
   - Modèle linéaire simple et interprétable.
   - Permet d'identifier des relations directes entre les features et la variable cible.

2. **Support Vector Machine (SVM)** :
   - Modèle basé sur la maximisation de la marge entre les classes.
   - Utilisation d'un noyau pour les problèmes non linéaires.

3. **K-Nearest Neighbors (KNN)** :
   - Basé sur la proximité entre les observations dans l'espace des features.
   - Particulièrement sensible à la normalisation et à l'échelle des données.

4. **Random Forest** :
   - Ensemble de modèles d'arbres de décision.
   - Fournit une robustesse accrue grâce à l'agrégation des prédictions des arbres.

5. **Gradient Boosting** :
   - Utilise un apprentissage séquentiel pour minimiser les erreurs des prédictions précédentes.
   - Excellente performance dans de nombreux cas.

6. **XGBoost** :
   - Variante optimisée de Gradient Boosting, particulièrement efficace en termes de temps et de performance.


**Entraînement des Modèles**

Les modèles ont été entraînés sur les données équilibrées après sur-échantillonnage de la classe minoritaire. La méthode de travail est la suivante :

1. **Entraîner des Modèles Individuels** :
   - Exemples de modèles spécifiques entraînés directement :

   .. code-block:: python

      # Entraîner un modèle Random Forest
      model_rf = RandomForestClassifier(random_state=42)
      model_rf.fit(X_train_balanced, y_train_balanced)

      # Entraîner un modèle de régression logistique
      model_lr = LogisticRegression(random_state=42)
      model_lr.fit(X_train_balanced, y_train_balanced)

      # Entraîner un modèle SVM
      model_svc = SVC(random_state=42)
      model_svc.fit(X_train_balanced, y_train_balanced)

2. **Boucle sur Différents Modèles** :
   - Une approche systématique a permis de tester plusieurs modèles :

   .. code-block:: python

      # Définir une liste de modèles à tester
      models = {
          'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
          'SVM': SVC(probability=True, random_state=42),
          'KNN': KNeighborsClassifier(),
          'Random Forest': RandomForestClassifier(random_state=42),
          'Gradient Boosting': GradientBoostingClassifier(random_state=42),
          'XGBoost': XGBClassifier(random_state=42)
      }

      # Entraîner chaque modèle
      for name, model in models.items():
          print(f"Training {name}...")
          model.fit(X_train_balanced, y_train_balanced)

**Résumé des Modèles**

Chaque modèle a été entraîné sur les données équilibrées, garantissant une représentation équitable des classes. Les paramètres par défaut ont été utilisés lors du premier essai. Par la suite, les modèles les plus performants ont été ajustés pour affiner leurs hyperparamètres (cette étape sera documentée dans la section suivante).

Les performances seront comparées pour choisir le modèle le plus adapté en fonction des métriques comme la précision, le rappel, la F1-score et l'AUC-ROC.

Modèle des Réseaux de Neurones Convolutifs (CNN)
------------------------------------------------

Dans cette section, nous expliquons le rôle des réseaux de neurones convolutifs (CNN) dans les tâches de classification d'images, suivie du processus de construction et d'entraînement d'un modèle CNN pour les données d'images.

---

**Description et Avantages**

Les CNN (Convolutional Neural Networks) sont un type de modèle d'apprentissage profond spécifiquement conçu pour traiter les données structurées sous forme d'images. Contrairement aux réseaux de neurones classiques, les CNN exploitent les propriétés spatiales des images à travers des couches de convolution et de pooling.

**Utilisation dans la Classification d'Images**

Dans un projet de classification d'images comme celui-ci, les CNN jouent un rôle crucial pour :

1. **L'extraction des caractéristiques visuelles** : Les couches convolutives identifient des motifs comme les bords, les textures ou des formes spécifiques dans les images.
2. **La différenciation entre classes** : En apprenant des caractéristiques propres à chaque catégorie, le modèle peut distinguer des images appartenant à différentes classes.
3. **La prise de décision** : À travers des couches entièrement connectées (Fully Connected Layers), les CNN peuvent interpréter les informations extraites et attribuer une classe à une image donnée.

---

**Construction et Entraînement d’un Modèle CNN**

**Architecture du Modèle**

Dans ce projet, nous avons conçu un CNN simple mais efficace, adapté aux besoins de la classification dans 4 catégories : Lesion, Hamstring, Entorse_Cheville et ACL.  
L'architecture comprend :
   - **Couches convolutives** : Pour l'extraction de caractéristiques (32, 64, et 128 filtres successivement).
   - **Couches de pooling (MaxPooling2D)** : Pour la réduction de la taille des caractéristiques tout en préservant les informations pertinentes.
   - **Couches Fully Connected (Dense)** : Pour combiner les caractéristiques extraites et produire des prédictions.
   - **Dropout** : Pour régulariser le modèle et éviter le surapprentissage.

**Résumé du modèle** :
   1. Trois couches convolutives successives, chacune suivie d'une couche MaxPooling.
   2. Une couche d'aplatissement (Flatten) pour transformer les caractéristiques extraites en un vecteur.
   3. Deux couches denses :
         - 128 neurones activés par `relu`.
         - Une couche de sortie avec 4 neurones activés par `softmax` (une pour chaque classe).

**Code de Construction du Modèle**

.. code-block:: python

   from tensorflow.keras import Sequential
   from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

   model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
      MaxPooling2D((2, 2)),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D((2, 2)),
      Conv2D(128, (3, 3), activation='relu'),
      MaxPooling2D((2, 2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(4, activation='softmax')  # 4 classes
   ])


**Entraînement et Validation**

Les données préparées (avec ImageDataGenerator) ont été utilisées pour entraîner le modèle. Le processus inclut :

   **Entraînement** : Ajustement des poids du modèle à l’aide des données d’entraînement.
   **Validation** : Évaluation du modèle après chaque époque pour surveiller ses performances.

.. code-block:: python
   history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator
   )