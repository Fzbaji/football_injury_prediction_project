import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image

# Chemin vers le modèle sauvegardé
MODEL_PATH = r'C:\Users\Dell\Desktop\Projet_IA\modelcnn.h5'

# Charger le modèle
@st.cache_resource
def load_trained_model(model_path):
    model = load_model(model_path)
    return model

# Fonction de prédiction
def predict_image(img, model, class_indices):
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_labels = list(class_indices.keys())
    
    return class_labels[predicted_class], prediction[0]

# Streamlit UI
st.title("Classification des Blessures des Joueurs")

# Charger le modèle
model = load_trained_model(MODEL_PATH)
class_indices = {'ACL': 0, 'Entorse_cheville1': 1, 'Lésion': 2, 'hamstring1': 3}

# Interface utilisateur
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Affichage de l'image téléchargée
    img = Image.open(uploaded_file)
    st.image(img, caption='Image téléchargée', use_column_width=True)
    
    # Prédiction
    predicted_class, probabilities = predict_image(img, model, class_indices)
    st.write(f"Classe prédite : *{predicted_class}*")
    
    # Afficher les probabilités
    st.write("Probabilités par classe :")
    for label, prob in zip(class_indices.keys(), probabilities):
        st.write(f"{label}: {prob:.2f}")

st.text("Développé avec ❤ par votre modèle AI")