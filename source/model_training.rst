Entraînement des Modèles
========================

Une étape clé de l’analyse a été de tester plusieurs algorithmes d’apprentissage supervisé pour prédire les risques de blessures musculaires. Les modèles suivants ont été entraînés en utilisant les données équilibrées après les étapes de prétraitement.

Algorithmes Utilisés
------------------------

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

Entraînement des Modèles
----------------------------

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

Résumé des Modèles
----------------------

Chaque modèle a été entraîné sur les données équilibrées, garantissant une représentation équitable des classes. Les paramètres par défaut ont été utilisés lors du premier essai. Par la suite, les modèles les plus performants ont été ajustés pour affiner leurs hyperparamètres (cette étape sera documentée dans la section suivante).

Les performances seront comparées pour choisir le modèle le plus adapté en fonction des métriques comme la précision, le rappel, la F1-score et l'AUC-ROC.

