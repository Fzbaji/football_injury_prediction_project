''' 
import streamlit as st
import pandas as pd
import joblib

# Chargement du modèle de prédiction et du scaler
model = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\modele_tab.h5')  # Remplacez par le chemin de votre modèle
scaler = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\scaler.pkl')  # Chargez le scaler sauvegardé

# Fonction pour prédire le risque de blessure
def predire_risque_blessure(data):
    # Transformation des données avec le scaler sauvegardé
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction

# Interface utilisateur avec Streamlit
st.title("Prédiction du Risque de Blessure des Joueurs de Football")

# Formulaire pour entrer les données du joueur
st.header("Entrez les informations du joueur")

# Saisie des données par l'utilisateur
player_age = st.number_input("Âge du joueur", min_value=16, max_value=50)
player_weight = st.number_input("Poids du joueur (kg)", min_value=40, max_value=150)
player_height = st.number_input("Taille du joueur (cm)", min_value=140, max_value=220)
previous_injuries = st.number_input("Nombre de blessures précédentes", min_value=0)
training_intensity = st.number_input("Intensité de l'entraînement (1 à 10)", min_value=1, max_value=10)
recovery_time = st.number_input("Temps de récupération estimé (jours)", min_value=0)

# Convertir les données entrées en DataFrame
data_joueur = pd.DataFrame({
    'Player_Age': [player_age],
    'Player_Weight': [player_weight],
    'Player_Height': [player_height],
    'Previous_Injuries': [previous_injuries],
    'Training_Intensity': [training_intensity],
    'Recovery_Time': [recovery_time]
})

# Prédire le risque de blessure quand l'utilisateur soumet les données
if st.button("Prédire le risque de blessure"):
    prediction = predire_risque_blessure(data_joueur)
    
    if prediction == 1:
        st.write("Le joueur est à risque de blessure.")
    else:
        st.write("Le joueur ne présente pas de risque immédiat de blessure.")

# Affichez les probabilités pour diagnostiquer
proba = model.predict_proba(data_joueur)
st.write(f"Probabilité de blessure : {proba[0][1]}")
st.write(f"Probabilité de pas de blessure : {proba[0][0]}")
'''

'''


import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Chargement du modèle et du scaler
model = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\modele2_tab.h5')
scaler = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\scaler.pkl')

# Fonction pour prédire le risque de blessure
def predire_risque_blessure(data):
    # Transformation des données avec le scaler sauvegardé
    try:
        data_scaled = scaler.transform(data)
    except ValueError as e:
        st.write(f"Erreur dans les données transformées : {e}")
        return None

    prediction = model.predict(data_scaled)
    probabilities = model.predict_proba(data_scaled)
    #prediction = model.predict(data)
    #probabilities = model.predict_proba(data)
    return prediction, probabilities

# Interface utilisateur
st.title("Prédiction du Risque de Blessure des Joueurs de Football")
st.header("Entrez les informations du joueur")

# Saisie des données utilisateur
player_age = st.number_input("Âge du joueur", min_value=16, max_value=50, step=1)
player_weight = st.number_input("Poids du joueur (kg)", min_value=40.0, max_value=150.0, step=1.0)
player_height = st.number_input("Taille du joueur (cm)", min_value=140.0, max_value=220.0, step=1.0)
previous_injuries = st.number_input("Nombre de blessures précédentes", min_value=0, step=1)
training_intensity = st.number_input("Intensité de l'entraînement (0 à 1)", min_value=0.0, max_value=1.0, step=0.001)
recovery_time = st.number_input("Temps de récupération estimé (jours)", min_value=0, step=1)

# Création du DataFrame utilisateur
data_joueur = pd.DataFrame({
    'Player_Age': [player_age],
    'Player_Weight': [player_weight],
    'Player_Height': [player_height],
    'Previous_Injuries': [previous_injuries],
    'Training_Intensity': [training_intensity],
    'Recovery_Time': [recovery_time]
})

# Prédiction
if st.button("Prédire le risque de blessure"):
    prediction, probabilities = predire_risque_blessure(data_joueur)
    if prediction is not None:
        st.write(f"Prédiction brute : {prediction[0]}")
        st.write(f"Probabilité de blessure : {probabilities[0][1]:.2f}")
        st.write(f"Probabilité de pas de blessure : {probabilities[0][0]:.2f}")

        if prediction[0] == 1:
            st.write("Le joueur est à risque de blessure.")
        else:
            st.write("Le joueur ne présente pas de risque immédiat de blessure.")
    else:
        st.write("Une erreur s'est produite lors de la prédiction.")
'''


'''
import streamlit as st
import pandas as pd
import joblib

# Chargement du modèle
model = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\modele2_tab.h5')
# Charger le scaler
scaler = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\scaler2.pkl')

# Fonction pour prédire avec scaling
def predire_risque_blessure(data):
    try:
        # Normaliser les données
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        probabilities = model.predict_proba(data_scaled)
        return prediction, probabilities
    except ValueError as e:
        st.write(f"Erreur dans la prédiction : {e}")
        return None, None

# Interface utilisateur
st.title("Prédiction du Risque de Blessure des Joueurs de Football")
st.header("Entrez les informations du joueur")

# Saisie des données utilisateur
player_age = st.number_input("Âge du joueur", min_value=16, max_value=50, step=1)
player_height = st.number_input("Taille du joueur (cm)", min_value=140.0, max_value=220.0, step=1.0)
player_weight = st.number_input("Poids du joueur (kg)", min_value=40.0, max_value=150.0, step=1.0)
'''
# Saisie des positions (une seule peut être active à la fois)
st.subheader("Sélectionnez la position principale du joueur (seule une position doit être activée) :")
positions = {
    'Centre-Back': 0,
    'Right Winger': 0,
    'Left Winger': 0,
    'Goalkeeper': 0,
    'Central Midfield': 0,
    'Right-Back': 0,
    'Left-Back': 0,
    'Defensive Midfield': 0,
    'Centre-Forward': 0,
    'Attacking Midfield': 0,
    'Left Midfield': 0,
    'Right Midfield': 0,
    'Second Striker': 0,
}

# L'utilisateur peut cocher une seule position
for position in positions.keys():
    positions[position] = st.checkbox(position)

# Vérifier qu'une seule position est activée
if sum(positions.values()) != 1:
    st.warning("Veuillez sélectionner une seule position.")
else:
    selected_position = [1 if positions[pos] else 0 for pos in positions]
'''
selected_position = st.multiselect(
    "Sélectionnez les positions du joueur :",
    options=[
        "Centre-Back", "Right Winger", "Left Winger", "Goalkeeper",
        "Central Midfield", "Right-Back", "Left-Back", "Defensive Midfield",
        "Centre-Forward", "Attacking Midfield", "Left Midfield", "Right Midfield",
        "Second Striker"
    ]
)

# Création du DataFrame utilisateur
data_joueur = pd.DataFrame({
    'Age': [player_age],
    'Height': [player_height],
    'Weight': [player_weight],
    'Centre-Back': [1 if 'Centre-Back' in selected_position else 0],
    'Right Winger': [1 if 'Right Winger' in selected_position else 0],
    'Left Winger': [1 if 'Left Winger' in selected_position else 0],
    'Goalkeeper': [1 if 'Goalkeeper' in selected_position else 0],
    'Central Midfield': [1 if 'Central Midfield' in selected_position else 0],
    'Right-Back': [1 if 'Right-Back' in selected_position else 0],
    'Left-Back': [1 if 'Left-Back' in selected_position else 0],
    'Defensive Midfield': [1 if 'Defensive Midfield' in selected_position else 0],
    'Centre-Forward': [1 if 'Centre-Forward' in selected_position else 0],
    'Attacking Midfield': [1 if 'Attacking Midfield' in selected_position else 0],
    'Left Midfield': [1 if 'Left Midfield' in selected_position else 0],
    'Right Midfield': [1 if 'Right Midfield' in selected_position else 0],
    'Second Striker': [1 if 'Second Striker' in selected_position else 0]
})

# Vérification du DataFrame pour les colonnes sélectionnées
st.write("Données utilisateur pour prédiction :")
st.write(data_joueur)



# Prédiction
if st.button("Prédire le risque de blessure"):
    prediction, probabilities = predire_risque_blessure(data_joueur)
    if prediction is not None:
        st.write(f"Prédiction brute : {prediction[0]}")
        st.write(f"Probabilité de blessure : {probabilities[0][1]:.2f}")
        st.write(f"Probabilité de pas de blessure : {probabilities[0][0]:.2f}")

        if prediction[0] == 1:
            st.write("Le joueur est à risque de blessure.")
        else:
            st.write("Le joueur ne présente pas de risque immédiat de blessure.")
    else:
        st.write("Une erreur s'est produite lors de la prédiction.")
'''

'''
import streamlit as st
import pandas as pd
import joblib

# Chargement du modèle
model = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\modele3_tab.h5')
# Charger le scaler
scaler = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\scaler3.pkl')

# Fonction pour prédire avec scaling
def predire_risque_blessure(data):
    try:
        # Normaliser les données
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        probabilities = model.predict_proba(data_scaled)
        return prediction, probabilities
    except ValueError as e:
        st.write(f"Erreur dans la prédiction : {e}")
        return None, None

# Interface utilisateur
st.title("Prédiction du Risque de Blessure des Joueurs de Football")
st.header("Entrez les informations du joueur")

# Saisie des données utilisateur
fatigue = st.number_input("Fatigue (%)", min_value=0.0, max_value=100.0, step=0.1)
heures_entrainement = st.number_input("Heures d'entraînement", min_value=0, max_value=50, step=1)
heures_match = st.number_input("Heures de match ", min_value=0, max_value=20, step=1)
gravité = st.selectbox("Gravité de la blessure précédente", ["Aucune", "Légère", "Modérée", "Grave"])
jours_absence = st.number_input("Nombre de jours d'absence (historique)", min_value=0, max_value=365, step=1)

# Encodage des variables catégoriques
type_blessure = st.selectbox("Type de blessure", ["Aucune", "Ligamentaire", "Musculaire"])
mecanisme = st.selectbox("Mécanisme de blessure", ["Aucun", "Collision", "Course/sprint", "Fatigue/surcharge"])

# Transformation des colonnes catégoriques
gravité_map = {"Aucune": 0, "Légère": 1, "Modérée": 2, "Grave": 3}
type_blessure_map = {"Aucune": [1, 0], "Ligamentaire": [0, 1], "Musculaire": [0, 0]}  # One-Hot Encoding
mecanisme_map = {
    "Aucun": [0, 0, 0],
    "Collision": [1, 0, 0],
    "Course/sprint": [0, 1, 0],
    "Fatigue/surcharge": [0, 0, 1],
}

# Mapping des valeurs
gravité_encoded = gravité_map[gravité]
type_blessure_encoded = type_blessure_map[type_blessure]
mecanisme_encoded = mecanisme_map[mecanisme]

# Création du DataFrame utilisateur
data_joueur = pd.DataFrame({
    'Fatigue': [fatigue],
    'Heures_entrainement': [heures_entrainement],
    'Heures_match': [heures_match],
    'Gravité': [gravité_encoded],
    'Jours_absence': [jours_absence],
    'Type_blessure_Ligamentaire': [type_blessure_encoded[0]],
    'Type_blessure_Musculaire': [type_blessure_encoded[1]],
    'Mecanisme_Collision': [mecanisme_encoded[0]],
    'Mecanisme_Course/sprint': [mecanisme_encoded[1]],
    'Mecanisme_Fatigue/surcharge': [mecanisme_encoded[2]],
})

# Afficher les données utilisateur
st.write("Données utilisateur pour prédiction :")
st.write(data_joueur)

# Prédiction
if st.button("Prédire le risque de blessure"):
    prediction, probabilities = predire_risque_blessure(data_joueur)
    if prediction is not None:
        st.write(f"Prédiction brute : {prediction[0]}")
        st.write(f"Probabilité de blessure : {probabilities[0][1]:.2f}")
        st.write(f"Probabilité de pas de blessure : {probabilities[0][0]:.2f}")

        if prediction[0] == 1:
            st.write("Le joueur est à risque de blessure.")
        else:
            st.write("Le joueur ne présente pas de risque immédiat de blessure.")
    else:
        st.write("Une erreur s'est produite lors de la prédiction.")
'''

'''
import streamlit as st
import pandas as pd
import joblib

# Chargement du modèle
model = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\modele3_tab.h5')
# Charger le scaler
scaler = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\scaler3.pkl')

# Fonction pour prédire avec scaling
def predire_risque_blessure(data):
    try:
        # Normaliser les données
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        probabilities = model.predict_proba(data_scaled)
        return prediction, probabilities
    except ValueError as e:
        st.write(f"Erreur dans la prédiction : {e}")
        return None, None

# Interface utilisateur
st.title("Prédiction du Risque de Blessure des Joueurs de Football")
st.header("Entrez les informations du joueur")

# Saisie des données utilisateur
fatigue = st.number_input("Fatigue (%)", min_value=0.0, max_value=100.0, step=0.1)
heures_entrainement = st.number_input("Heures d'entraînement ", min_value=0, max_value=50, step=1)
heures_match = st.number_input("Heures de match", min_value=0, max_value=20, step=1)

# Encodage des variables catégoriques
mecanisme = st.selectbox("Mécanisme de blessure", ["Aucun", "Collision", "Course/sprint", "Fatigue/surcharge"])

# Transformation des colonnes catégoriques
mecanisme_map = {
    "Aucun": [0, 0, 0],
    "Collision": [1, 0, 0],
    "Course/sprint": [0, 1, 0],
    "Fatigue/surcharge": [0, 0, 1],
}

# Mapping des valeurs
mecanisme_encoded = mecanisme_map[mecanisme]

# Création du DataFrame utilisateur
data_joueur = pd.DataFrame({
    'Fatigue': [fatigue],
    'Heures_entrainement': [heures_entrainement],
    'Heures_match': [heures_match],
    'Mecanisme_Collision': [mecanisme_encoded[0]],
    'Mecanisme_Course/sprint': [mecanisme_encoded[1]],
    'Mecanisme_Fatigue/surcharge': [mecanisme_encoded[2]],
})

# Afficher les données utilisateur
st.write("Données utilisateur pour prédiction :")
st.write(data_joueur)

# Prédiction
if st.button("Prédire le risque de blessure"):
    prediction, probabilities = predire_risque_blessure(data_joueur)
    if prediction is not None:
        st.write(f"Prédiction brute : {prediction[0]}")
        st.write(f"Probabilité de blessure : {probabilities[0][1]:.2f}")
        st.write(f"Probabilité de pas de blessure : {probabilities[0][0]:.2f}")

        if prediction[0] == 1:
            st.error("Le joueur est à risque de blessure.")
        else:
            st.success("Le joueur ne présente pas de risque immédiat de blessure.")
    else:
        st.write("Une erreur s'est produite lors de la prédiction.")
'''
'''
#partie seule
# Prédiction
if st.button("Prédire"):
    prediction = model.predict(data_input)[0]  # Prévoir 0 ou 1
    proba = model.predict_proba(data_input)[0]  # Probabilités associées

    if prediction == 1:
        st.error(f"Le joueur est à risque de blessure {proba[1]:.2%}")
    else:
        st.success(f"Le joueur ne présente pas de risque immédiat de blessure. {proba[0]:.2%}")
'''

'''
##IMPORTANT##
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Charger le modèle pré-entraîné
model = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\modele4_tab.h5')

# Titre de l'application
#st.title("Prédiction des Blessures Musculaires")

## Interface utilisateur
st.title("Prédiction du Risque de Blessure des Joueurs de Football")
st.header("Entrez les informations du joueur")

# Formulaire pour entrer les données utilisateur
st.sidebar.header("Paramètres d'entrée")
minutes_jouees = st.sidebar.number_input("Minutes jouées", min_value=0, step=1, value=1000)
fatigue = st.sidebar.slider("Fatigue (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
heures_entrainement = st.sidebar.number_input("Heures d’entraînement", min_value=0, step=1, value=20)
age = st.sidebar.number_input("Âge", min_value=0, step=1, value=25)
historique_blessures = st.sidebar.selectbox("Historique de blessures musculaires", [0, 1])
contact_sans_contact = st.sidebar.selectbox("Sans contact physique ?", ["Oui", "Non"])

# Conversion du champ 'Sans contact physique ?' en binaire
contact_physique_encoded = 1 if contact_sans_contact == "Oui" else 0

# Préparer les données sous forme de dataframe
data_input = pd.DataFrame({
    "Minutes jouées": [minutes_jouees],
    "Fatigue (%)": [fatigue],
    "Heures d’entraînement": [heures_entrainement],
    "Âge": [age],
    "Historique de blessures musculaires": [historique_blessures],
    "Contact physique_Sans contact": [bool(contact_physique_encoded)]
})

st.write("### Données entrées :")
st.write(data_input)

# Fonction pour prédire avec scaling
def predire_risque_blessure(data):
    try:
        prediction = model.predict(data)
        probabilities = model.predict_proba(data)
        return prediction, probabilities
    except ValueError as e:
        st.write(f"Erreur dans la prédiction : {e}")
        return None, None
    

# Prédiction
if st.button("Prédire le risque de blessure"):
    prediction, probabilities = predire_risque_blessure(data_input)
    if prediction is not None:
        st.write(f"Prédiction brute : {prediction[0]}")
        st.write(f"Probabilité de blessure : {probabilities[0][1]:.2f}")
        st.write(f"Probabilité de pas de blessure : {probabilities[0][0]:.2f}")

        if prediction[0] == 1:
            st.error("Le joueur est à risque de blessure.")
        else:
            st.success("Le joueur ne présente pas de risque immédiat de blessure.")
    else:
        st.write("Une erreur s'est produite lors de la prédiction.")
'''

'''
#aumentation_ACL

import albumentations as A
import cv2
import os

augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5)
])

def augment_class(input_dir, output_dir, target_count):
    os.makedirs(output_dir, exist_ok=True)
    images = os.listdir(input_dir)
    current_count = len(images)
    
    for i in range(target_count - current_count):
        img_name = images[i % current_count]  # Cycler les images existantes
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        
        # Augmenter et sauvegarder
        augmented = augmentation(image=image)['image']
        aug_img_name = f"aug_{i}_{img_name}"
        cv2.imwrite(os.path.join(output_dir, aug_img_name), augmented)

# Exemple : augmenter la classe ACL
augment_class(r"C:\Users\Dell\Desktop\Projet_IA\ACL", r"C:\Users\Dell\Desktop\Projet_IA\ACL_augmented", target_count=70)

'''

'''

# augmentation de toute la data

import albumentations as A
import cv2
import os

# Définir les transformations pour l'augmentation
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=5, p=0.5)
])

# Fonction pour augmenter une classe
def augment_class(input_dir, output_dir, target_count):
    os.makedirs(output_dir, exist_ok=True)  # Créer le répertoire de sortie s'il n'existe pas
    images = os.listdir(input_dir)  # Liste des images dans le répertoire source
    current_count = len(images)

    # Vérification : au moins une image doit être présente
    if current_count == 0:
        print(f"Erreur : Aucun fichier trouvé dans {input_dir}.")
        return

    for i in range(target_count - current_count):
        img_name = images[i % current_count]  # Cycler les images existantes
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        # Vérifier si l'image a été chargée correctement
        if image is None:
            print(f"Erreur : Impossible de lire l'image {img_path}. Elle sera ignorée.")
            continue

        # Appliquer les augmentations et sauvegarder l'image
        augmented = augmentation(image=image)['image']
        aug_img_name = f"aug_{i}_{img_name}"  # Créer un nouveau nom pour l'image augmentée
        aug_img_path = os.path.join(output_dir, aug_img_name)
        cv2.imwrite(aug_img_path, augmented)

    print(f"Augmentation terminée pour le dossier {input_dir}. Total images : {target_count}.")

# Définir les classes et leurs dossiers
classes = ["ACL", "Entorse_Cheville", "Hamstring", "Lésion"]
base_input_dir = r"C:\Users\Dell\Desktop\Projet_IA\data_images"
base_output_dir = r"C:\Users\Dell\Desktop\Projet_IA\data_augmented"

# Boucler sur les classes et augmenter les données
for class_name in classes:
    input_dir = os.path.join(base_input_dir, class_name)  # Chemin du répertoire d'entrée
    output_dir = os.path.join(base_output_dir, class_name)  # Chemin du répertoire de sortie
    augment_class(input_dir, output_dir, target_count=200)  # Augmenter à 200 images par classe

print("Augmentation terminée pour toutes les classes.")
'''
