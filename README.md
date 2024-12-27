<<<<<<< HEAD
# football-injury-prediction

---

Ce projet ambitionne de développer une application basée sur l’apprentissage automatique pour prédire et gérer efficacement les blessures. L’application envisagée comportera deux fonctionnalités principales :

- **Prédiction du risque de blessure** : Analyse des données de performance des joueurs (comme le temps de jeu, les séances d’entraînement, et l’historique médical) afin d’évaluer les risques avant qu’ils ne deviennent critiques.
- **Évaluation de la gravité et estimation du rétablissement** : Exploitation d’images pour déterminer la gravité des blessures diagnostiquées et estimer les délais de retour au jeu.



## Pipeline du Projet

1. **Collecte et Nettoyage des Données**
   - Génération de données synthétiques basées sur des variables clés liées aux risques de blessures musculaires.
   - Nettoyage, encodage et gestion des déséquilibres dans les données.
   - Collecte manuelle et nettoyage des images de blessures.

2. **Développement et Validation des Modèles d'Apprentissage Automatique**
   - Entraînement et validation de plusieurs modèles :
     - **Algorithmes traditionnels** pour la prédiction tabulaire : Random Forest (meilleur modèle), Régression Logistique, SVM, etc.
     - **CNN (Convolutional Neural Networks)** pour l’analyse d’images et l’évaluation de la gravité des blessures.
   - Validation des modèles sur des données réelles et comparaison par métriques (Accuracy, F1-Score, ROC-AUC).

3. **Interface Utilisateur Conviviale**
   - Création d'une application avec **Streamlit** pour :
     - Prédiction du risque de blessures musculaires à partir des données tabulaires.
     - Analyse d’images pour évaluer la gravité des blessures et estimer les délais de récupération.
     - Chargement et manipulation simplifiée des données utilisateur.

---


## Installation des Dépendances

Pour installer les bibliothèques nécessaires, et lancer l'application streamlit utilisez la commande suivante dans votre terminal :

```bash
pip install -r requirements.txt
 
## Lancez l'application streamlit

streamlit run strapp.py


=======
# football_injury_prediction_c
Projet de prédiction des blessures, et recommandation pour les joueurs de football
>>>>>>> 01ac5cb4d3a057e188324a81d71f73df3ccd347e
