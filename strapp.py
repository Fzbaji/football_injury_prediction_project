import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf

# Charger les modèles pré-entraînés
tabular_model = joblib.load(r'C:\Users\Dell\Desktop\Projet_IA\modele4_tab.h5')  # Modèle tabulaire
image_model = tf.keras.models.load_model(r'C:\Users\Dell\Desktop\Projet_IA\modelcnn.h5')  # Modèle CNN pour la classification des blessures

# Définir une fonction pour prédire avec le modèle CNN
def predire_type_blessure(image):
    # Pré-traiter l'image
    image = image.resize((150, 150))  # Assurez-vous que la taille correspond à l'entrée du modèle CNN
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Ajouter une dimension batch
    prediction = image_model.predict(image_array)
    return prediction

# Mapper les classes à des descriptions, gravités et durées de récupération
class_mapping = {
    0: ("Blessure ACL", "Très grave", "6-9 mois"),
    1: ("Entorse de la cheville", "Légère", "2-6 semaines"),
    2: ("Lésion", "Grave", "3-6 mois"),
    3: ("Blessure aux Ischio-jambiers (Hamstring)", "Moyenne", "4-8 semaines"),
}

# Définir la fonction pour afficher les prédictions tabulaires
def interface_tabulaire():
    st.title("Prédiction des Blessures et Recommandations pour les Joueurs de Football")
    st.header("Entrez les informations du joueur")

    # Formulaire d'entrée pour l'utilisateur
    st.sidebar.header("Paramètres d'entrée")
    minutes_jouees = st.sidebar.number_input("Minutes jouées", min_value=0, step=1, value=1000)
    fatigue = st.sidebar.slider("Fatigue (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    heures_entrainement = st.sidebar.number_input("Heures d’entraînement", min_value=0, step=1, value=20)
    age = st.sidebar.number_input("Âge", min_value=0, step=1, value=25)
    historique_blessures = st.sidebar.selectbox("Historique de blessures musculaires", [0, 1])
    contact_sans_contact = st.sidebar.selectbox("Sans contact physique ?", ["Oui", "Non"])

    # Convertir les entrées en données compatibles
    contact_physique_encoded = 1 if contact_sans_contact == "Oui" else 0
    data_input = pd.DataFrame({
        "Minutes jouées": [minutes_jouees],
        "Fatigue (%)": [fatigue],
        "Heures d’entraînement": [heures_entrainement],
        "Âge": [age],
        "Historique de blessures musculaires": [historique_blessures],
        "Contact physique_Sans contact": [bool(contact_physique_encoded)],
    })

    st.write("### Données entrées :")
    st.write(data_input)

    # Prédictions
    if st.button("Prédire le risque de blessure"):
        try:
            prediction = tabular_model.predict(data_input)
            probabilities = tabular_model.predict_proba(data_input)

            st.write(f"Prédiction brute : {prediction[0]}")
            st.write(f"Probabilité de blessure : {probabilities[0][1]:.2f}")
            st.write(f"Probabilité de pas de blessure : {probabilities[0][0]:.2f}")

            if prediction[0] == 1:
                st.error("Le joueur est à risque de blessure.")
            else:
                st.success("Le joueur ne présente pas de risque immédiat de blessure.")
        except Exception as e:
            st.write(f"Erreur dans la prédiction : {e}")

# Définir la fonction pour afficher les prédictions basées sur les images
def interface_images():
    st.title("Prédiction du type de la blessure, sa gravité, et recommandation pour temps de rétablissement")
    st.header("Téléchargez une image de la blessure")

    # Téléchargement de l'image
    uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Charger l'image
        image = Image.open(uploaded_file)
        st.image(image, caption="Image téléchargée", use_column_width=True)

        if st.button("Prédire le type de blessure"):
            prediction = predire_type_blessure(image)
            predicted_class = np.argmax(prediction)  # Obtenir la classe prédite
            blessure, gravité, délai = class_mapping[predicted_class]

            st.write(f"Type de blessure prédite : {blessure}")
            st.write(f"Gravité : {gravité}")
            st.write(f"Temps de rétablissement estimé : {délai}")

# Ajout d'une barre de navigation
st.sidebar.title("Menu de Navigation")
option = st.sidebar.radio("Choisissez une section :", ["Prédire le risque de blessure", "Prédiction du type de la blessure, gravité et recommandation"])

if option == "Prédire le risque de blessure":
    interface_tabulaire()
elif option == "Prédiction du type de la blessure, gravité et recommandation":
    interface_images()