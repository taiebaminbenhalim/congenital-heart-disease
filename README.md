# 🫀 Congenital Heart Disease — Classification binaire (Normal vs Malade)

Classification d'images d'échocardiographie pour détecter la présence d'une cardiopathie congénitale, avec prétraitement **CLAHE**, un **CNN from scratch** (baseline) et du **transfer learning avec ResNet18** (PyTorch).

> ⚠️ **Projet éducatif / de recherche.** Ce modèle n'est pas un dispositif médical et ne doit en aucun cas être utilisé pour établir un diagnostic.

---

## 📌 Présentation

Le dataset original contient 4 classes (ASD, Normal, PDA, VSD). Le problème est transformé en **classification binaire** :

| Classe | Source | Images |
|---|---|---|
| **Normal** (0) | Normal | 208 |
| **Malade** (1) | ASD (194) + PDA (216) + VSD (210) | 620 |
| **Total** | | 828 |

Le dataset est **déséquilibré** (1 Normal pour 3 Malades), d'où l'usage de métriques adaptées (F1-score, ROC-AUC) et de techniques anti-déséquilibre.

## 🛠️ Pipeline

1. **Prétraitement CLAHE** (OpenCV) sur le canal L de l'espace LAB pour améliorer le contraste local (`clipLimit=2.0`, tuiles 8×8)
2. **Dataset PyTorch personnalisé** (`BinaryHeartDataset`) regroupant ASD + PDA + VSD → Malade
3. **Split reproductible** 80 / 10 / 10 (train / val / test, seed = 42)
4. **Gestion du déséquilibre**
   - `WeightedRandomSampler` (batches ≈ 50/50)
   - `CrossEntropyLoss` pondérée + Label Smoothing (0.1)
   - Data augmentation (flip, rotation, affine, color jitter, flou gaussien)
5. **Modèles**
   - **CNN personnalisé** : 4 blocs Conv-BN-ReLU-MaxPool-Dropout2d (32→256 filtres), ~4,6 M de paramètres
   - **ResNet18 pré-entraîné (ImageNet)** : tout gelé sauf `layer4` + `fc` (`Dropout(0.5) → Linear(512, 2)`)
6. **Entraînement** : AdamW, `ReduceLROnPlateau`, gradient clipping (1.0), Early Stopping (patience = 7, max 40 époques)
7. **Évaluation** : Accuracy, Precision, Recall, F1-score, ROC-AUC, matrice de confusion

## 📊 Résultats (set de test — 84 images)

| Métrique | CNN (from scratch) | ResNet18 (transfer learning) |
|---|---|---|
| Accuracy | 54,76 % | **83,33 %** |
| Precision | **96,67 %** | 93,33 % |
| Recall (Malade) | 43,94 % | **84,85 %** |
| F1-score | 60,42 % | **88,89 %** |
| ROC-AUC | 0,7685 | **0,9369** |

Matrice de confusion ResNet18 : **TN = 14 · FP = 4 · FN = 10 · TP = 56**

➡️ Le transfer learning (ResNet18 + CLAHE) surpasse nettement le CNN entraîné from scratch. Les faux négatifs (10 malades non détectés) restent le point critique pour un usage médical.

## 📁 Structure du projet

```
congenital-heart-disease/
├── classification_binaire_CLAHE_CNN_ResNet.ipynb   # Notebook principal
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Installation et utilisation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<votre-utilisateur>/congenital-heart-disease.git
cd congenital-heart-disease
```

### 2. Installer les dépendances

```bash
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Accéder au dataset

Le dataset n'est **pas inclus** dans ce dépôt. Il est téléchargé via `kagglehub` depuis la compétition Kaggle `congenital-heart-disease`.

1. Créez un compte [Kaggle](https://www.kaggle.com) et acceptez les règles de la compétition
2. Authentifiez-vous (ne mettez **jamais** votre token dans le code ou dans Git) :

```python
import kagglehub
kagglehub.login()
```

### 4. Lancer le notebook

```bash
jupyter notebook classification_binaire_CLAHE_CNN_ResNet.ipynb
```

Le notebook peut aussi être exécuté sur Google Colab ou Kaggle (GPU recommandé, bien que le notebook ait été exécuté sur CPU).

## 🔮 Prédire une nouvelle image

Une fonction `predire_binaire(chemin_image, model, class_names, device)` est fournie en fin de notebook : elle affiche l'image, la classe prédite et les probabilités Normal / Malade.

## 🧰 Technologies

Python · PyTorch · Torchvision · OpenCV · scikit-learn · NumPy · Matplotlib · Seaborn · KaggleHub

## 🔭 Pistes d'amélioration

- Valider sur un jeu de test externe et par patient (éviter toute fuite de données entre train/test)
- Ajuster le seuil de décision pour maximiser le recall sur la classe Malade
- Essayer d'autres architectures (ResNet50, EfficientNet) et le fine-tuning complet
- Revenir à la classification multi-classes (ASD / PDA / VSD / Normal)
- Ajouter de l'explicabilité (Grad-CAM)

## 📄 Licence

À définir (ex. MIT). Le dataset reste soumis aux conditions d'utilisation de Kaggle / de ses auteurs.

## 👤 Auteur

**Tayeb** — [GitHub](https://github.com/<votre-utilisateur>)
