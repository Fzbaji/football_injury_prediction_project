import requests

def download_model(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Modèle téléchargé avec succès dans {output_path}")
    else:
        print("Erreur lors du téléchargement du modèle :", response.status_code)

if __name__ == "__main__":
    # Remplacez par le lien direct ou généré pour votre fichier Google Drive
    MODEL_URL = "https://drive.google.com/uc?id=ID_FICHIER_DRIVE&export=download"
    OUTPUT_PATH = "model_resnet_finetuned.h5"

    download_model(MODEL_URL, OUTPUT_PATH)
