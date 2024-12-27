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
classes = ["ACL", "Entorse_Cheville", "Hamstring", "Lesion"]
base_input_dir = r"C:\Users\Dell\Desktop\Projet_IA\data_images"
base_output_dir = r"C:\Users\Dell\Desktop\Projet_IA\data_augmented"

# Boucler sur les classes et augmenter les données
for class_name in classes:
    input_dir = os.path.join(base_input_dir, class_name)  # Chemin du répertoire d'entrée
    output_dir = os.path.join(base_output_dir, class_name)  # Chemin du répertoire de sortie
    augment_class(input_dir, output_dir, target_count=200)  # Augmenter à 200 images par classe

print("Augmentation terminée pour toutes les classes.")