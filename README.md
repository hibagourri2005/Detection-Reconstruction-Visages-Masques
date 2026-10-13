# 😷 Masked Face — Détection de masque et démasquage de visages

Projet de vision par ordinateur (Deep Learning) réalisé par **Gourri ,Foudal et Bellafrouh** — filière **IID2**.

Le projet comporte deux parties :

1. **Détection de masque** : savoir si une personne porte un masque sur une image (classification *mask / no_mask*).
2. **Démasquage** : reconstruire le visage complet à partir d'une image de visage masqué (avec **Pix2Pix**).

---

## 📁 Structure du dépôt

```
masked-face-project/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    ├── 01_approche1_classificateur_couleur_finetuning_yolo.ipynb
    ├── 02_approche2_yolov8_vgg16_vs_resnet50.ipynb
    ├── 03_pix2pix_demasquage.ipynb
    └── 04_architectures_modeles.ipynb
```

| Notebook | Contenu |
|---|---|
| `01_approche1...` | **Approche 1** : YOLOv8 (détection du visage) + classificateur basé sur la couleur (HSV), puis fine-tuning de YOLOv8-cls. Tests sur images, webcam et vidéo. |
| `02_approche2...` | **Approche 2** : YOLOv8 (extraction des visages) + CNN en transfer learning (**VGG16 vs ResNet50**). Pipeline complet et évaluation. |
| `03_pix2pix...` | **Démasquage** avec Pix2Pix (générateur U-Net + discriminateur PatchGAN) sur CelebA masqué. |
| `04_architectures...` | Visualisation des architectures utilisées (VGG16, U-Net, PatchGAN). |

---

## 🧪 Méthodes

### Détection de masque
- **Approche 1 (baseline)** : YOLOv8n détecte le visage, puis une heuristique de couleur (HSV) décide s'il y a un masque. Limite : ne fonctionne que pour certaines couleurs de masque.
- **Approche 2 (retenue)** : YOLOv8n extrait les visages, puis un **CNN en transfer learning** (VGG16 ou ResNet50, entraînement en 2 phases : tête puis fine-tuning) classe *mask / no_mask*.

### Démasquage
- **Pix2Pix** : générateur **U-Net** (≈ 54,4 M de paramètres) + discriminateur **PatchGAN**, entraîné 40 epochs.

---

## 📊 Résultats

### Classification (jeu de test, 142 images)

| Modèle | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **VGG16 (fine-tuné)** | **99.29 %** | 99.57 % | 99.01 % | **99.29 %** |
| ResNet50 (fine-tuné) | 94.22 % | 96.19 % | 92.04 % | 94.07 % |

### Pipeline complet sur images réelles (Dataset 2, 200 images)

| | Approche 1 (HSV) | Approche 2 (YOLO + VGG16) |
|---|---|---|
| Accuracy | 56.50 % | **99.75 %** |
| Precision | 84.21 % | **100 %** |
| Recall | 16.00 % | **99.50 %** |
| F1 | 26.89 % | **99.75 %** |

➡️ **YOLO + VGG16** est le meilleur système : le transfer learning apprend les vraies caractéristiques du masque, indépendamment de sa couleur.

### Démasquage (Pix2Pix, 1 000 images de test)

| Métrique | Valeur |
|---|---|
| SSIM | **0.8545** (± 0.0099) |
| PSNR | **25.07 dB** (± 0.64) |

---

## 🗂️ Données

Les datasets ne sont **pas inclus** dans ce dépôt (trop volumineux).

- **Dataset 1** : images de visages `Masked` (15 442) / `NonMasked` (15 409).
- **Dataset 2** : images complètes `with_mask` / `without_mask` (utilisées pour l'évaluation en conditions réelles).
- **Démasquage** : *splitted-masked-celeba* (Part1) — 8 000 paires pour l'entraînement, 1 000 pour la validation, 1 000 pour le test.

> 📌 Ajoutez ici les liens de téléchargement de vos datasets (Kaggle, etc.).

---

## ⚙️ Installation et utilisation

```bash
git clone https://github.com/<votre-utilisateur>/<nom-du-repo>.git
cd <nom-du-repo>
pip install -r requirements.txt
jupyter notebook
```

Ouvrez ensuite les notebooks dans le dossier `notebooks/` (de préférence sur **Google Colab** ou Kaggle avec GPU) et adaptez les chemins vers vos datasets dans la section « Configuration des chemins ».

---

## 🛠️ Technologies

Python · TensorFlow / Keras · PyTorch · Ultralytics YOLOv8 · OpenCV · scikit-learn · Matplotlib · Seaborn

---

## 👥 Auteurs
- **Gourri Hiba**
- **Bellafrouh Hajar**
- **Foudal Ihsane**


Filière IID2
