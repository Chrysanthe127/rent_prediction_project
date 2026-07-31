# 🏠 Système de prédiction de loyer — MEDIABOX Burundi

---

## 📌 Aperçu du projet

Ce projet, réalisé pour **MEDIABOX Burundi**, vise à construire un pipeline complet de Machine Learning (de l'EDA au déploiement) pour estimer le loyer mensuel d'une maison à Bujumbura à partir de ses caractéristiques physiques et de sa localisation.

L'objectif est de remplacer les estimations subjectives par un outil fiable, standardisé et accessible aux non-techniciens via une application **Streamlit**.

---

## 🎯 Problème métier

| Problème | Solution | Valeur Business |
|----------|----------|-----------------|
| Estimation des loyers incohérente entre agents et quartiers | Modèle de régression basé sur les données historiques | Uniformisation des estimations |
| Temps de mise sur le marché trop long | Estimation instantanée via l'application | Réduction du temps de commercialisation |
| Outil complexe pour les non-experts | Interface Streamlit intuitive | Accessible sans compétence technique |

---

## 📊 Description du dataset

**Fichier** : `rent_prediction.csv`  
**Taille** : 510 lignes, 12 colonnes  
**Données manquantes** : ~4,9 % (25 lignes) réparties uniformément, traitées lors de l'EDA

| Colonne | Type | Description |
|---------|------|-------------|
| `IdentifiantMaison` | int | ID unique (exclu du modèle) |
| `Chambres` | float | Nombre de chambres |
| `Salon` | catégoriel | Oui/Non |
| `SalleDeBainInterieure` | catégoriel | Oui/Non |
| `Parking` | catégoriel | Oui/Non |
| `Meuble` | catégoriel | Oui/Non |
| `Jardin` | catégoriel | Oui/Non |
| `Superficie_m2` | float | Surface en m² |
| `DistanceRoute_m` | float | Distance à la route principale (m) |
| `Quartier` | catégoriel | Quartier de Bujumbura |
| `AgeMaison` | float | Âge de la maison (années) |
| `LoyerMensuel_BIF` | int | **Variable Cible (Régression)** |

---

## ⚙️ Pipeline technique détaillé

Le projet suit un pipeline structuré en **9 phases** :

### Phase 1 — Business Understanding
- Définition du problème métier, des parties prenantes.
- Identification de la variable cible : `LoyerMensuel_BIF`.
- Contraintes : données manquantes (4,9 %), taille réduite (510 lignes), variable quartier catégorielle.

### Phase 2 — Exploratory Data Analysis (EDA)
- **Cible** : Asymétrie positive (skewness = 0.59), plafonnement artificiel à 2 500 000 BIF.
- **Numériques** : Multicolinéarité entre `Chambres` et `Superficie_m2` (corrélation ≈ 0.98). `DistanceRoute_m` et `AgeMaison` ont une corrélation quasi nulle avec la cible (≈ 0.02).
- **Catégorielles** : Le quartier a un impact prédominant. Les commodités (`Salon`, `Jardin`) augmentent le loyer.

### Phase 3 — Data Preparation
- Suppression des 25 lignes contenant des NaN (MNAR).
- Suppression des lignes plafonnées à 2.5M BIF (valeurs aberrantes suspectes).
- Suppression de `Chambres` pour éliminer la multicolinéarité.

### Phase 4 — Feature Engineering
- **`Confort_Score`** : Somme des 5 commodités (Salon, SDB, Parking, Meuble, Jardin). Score de 0 à 5. Suppression des 5 colonnes binaires.
- **`Chambres_par_Superficie`** : Ratio de densité (`Chambres / Superficie`).
- **Regroupement des Quartiers** : Fusion des quartiers avec < 25 occurrences dans la catégorie "Autres".
- **Transformation de la cible** : `np.log1p()` sur `LoyerMensuel_BIF` pour normaliser la distribution.
- **Transformations logarithmiques** : `DistanceLog = log(1 + DistanceRoute_m)`, `AgeMaisonLog = log(1 + AgeMaison)`.
- **Variables d'interaction** : `Confort_Surface`, `Confort_Chambres`, `Age_Surface`.

### Phase 5 — Baseline Models
Références objectives avant toute complexité :

| Modèle | MAE (BIF) | R² |
|--------|-----------|-----|
| Dummy Regressor (Moyenne) | 125 230 | 0.000 |
| Dummy Regressor (Médiane) | 120 100 | -0.012 |
| Linear Regression | 81 200 | 0.765 |

### Phase 6 — Model Experiments
Comparaison rigoureuse via **5-Fold Cross-Validation** :

| Modèle | MAE (BIF) | RMSE (BIF) | R² |
|--------|-----------|------------|-----|
| Ridge (α=1.0) | 80 500 | 108 200 | 0.772 |
| Random Forest (n=100) | 64 800 | 91 800 | 0.835 |
| **Gradient Boosting (n=300)** | **64 200** | **90 500** | **0.842** |

### Phase 7 — Hyperparameter Tuning
Optimisation du modèle **Gradient Boosting** via `GridSearchCV`.

| Paramètre | Valeur optimale |
|-----------|-----------------|
| `learning_rate` | 0.05 |
| `n_estimators` | 300 |
| `max_depth` | 5 |
| `min_samples_split` | 2 |

### Phase 8 — Feature Importance
- Extraction des **5 features les plus importantes** : `Superficie_m2`, `Confort_Score`, `Quartier_Kiriri`, `Quartier_Ngagara`, `DistanceLog`.
- Comparaison modèle complet vs modèle réduit (Top 5) :

| Modèle | MAE (BIF) | R² |
|--------|-----------|-----|
| Modèle complet (31 features) | 64 200 | 0.842 |
| Modèle réduit (5 features) | 67 100 | 0.826 |

**Décision** : Le modèle réduit est recommandé pour le déploiement (simplification massive pour une perte minime de performance).

### Phase 9 — Deployment (Streamlit)
- Interface web interactive développée avec **Streamlit** dans **VS Code**.
- L'utilisateur saisit les 10 variables brutes.
- L'application recalcule automatiquement : `Confort_Score`, `Chambres_par_Superficie`, transformations log, one-hot encoding.
- Prédiction retournée en BIF via `np.expm1()`.
- Déploiement sur **Streamlit Cloud** : [https://rentpredictionproject.streamlit.app](https://rentpredictionproject.streamlit.app)

---

## 📁 Structure du dépôt
rent_prediction_project/
│
├── 📂 data/                                 # Données du projet
│   └── 📄 rent_prediction.csv               # Dataset brut (510 lignes, 12 colonnes)
│
├── 📂 notebooks/                            # Notebooks d'analyse et de modélisation
│   ├── 📓 01_EDA.ipynb                      # Analyse exploratoire des données
│   ├── 📓 02_Data_Preparation.ipynb         # Nettoyage et prétraitement
│   ├── 📓 03_Feature_Engineering.ipynb      # Création des variables dérivées
│   ├── 📓 04_Baseline_Model.ipynb           # Modèles de référence
│   ├── 📓 05_Model_Experiments.ipynb        # Comparaison des algorithmes
│   ├── 📓 06_Hyperparameter_Tuning.ipynb    # Optimisation des hyperparamètres
│   └── 📓 07_Feature_Importance.ipynb       # Analyse d'importance des features
│
├── 📂 streamlit/                            # Application de déploiement
│   └── 🖥️ app.py                            # Interface utilisateur Streamlit
│
├── 📂 models/                               # Modèles entraînés
│   └── 🧠 best_model_tuned.pkl              # Modèle final (Gradient Boosting)
│
├── 📂 reports/                              # Documentation et rapports
│   ├── 📄 Business_Understanding.md         # Compréhension du problème métier
│   ├── 📄 EDA_Report.pdf                    # Rapport d'analyse exploratoire
│   └── 📄 Feature_Engineering_Report.md     # Rapport du feature engineering
│
├── 📂 figures/                              # Visualisations exportées
│   └── 📊 feature_importance.png            # Graphique des features importantes
│
├── 📂 logs/                                 # Suivi des expériences
│   └── 📋 experiment_log.csv                # Journal des performances des modèles
│
├── 📄 requirements.txt                      # Dépendances Python (versions figées)
├── 📄 README.md                             # Documentation principale du projet
└── 📄 .gitignore                            # Fichiers et dossiers exclus de Git

text

---

## 🛠️ Installation & Dépendances

### 1. Télécharger le projet depuis Colab

Tous les notebooks ont été développés et exécutés dans **Google Colab**. Une fois le travail terminé, le projet a été téléchargé depuis Colab sous forme de dossier (`rent_prediction_project`).

### 2. Créer un environnement virtuel local

**Windows :**
```bash
python -m venv venv
venv\Scripts\activate
macOS/Linux :

bash
python3 -m venv venv
source venv/bin/activate
3. Installer les dépendances
bash
pip install -r requirements.txt
Fichier requirements.txt
txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.6.1
joblib==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
🚀 Exécution du projet
1. Notebooks – Google Colab
Les notebooks sont disponibles dans le dossier notebooks/ et peuvent être rouverts dans Colab :

Ouvrez Google Colab

Téléversez les fichiers .ipynb depuis le dossier notebooks/ :

01_EDA.ipynb

02_Data_Preparation.ipynb

03_Feature_Engineering.ipynb

04_Baseline_Model.ipynb

05_Model_Experiments.ipynb

06_Hyperparameter_Tuning.ipynb

07_Feature_Importance.ipynb

Exécutez les cellules dans l'ordre.

Astuce : Pour charger les données dans Colab, vous pouvez monter Google Drive ou cloner le dépôt :

python
from google.colab import drive
drive.mount('/content/drive')
2. Application Streamlit (VS Code)
L'application a été développée dans VS Code. Pour la lancer en local :

bash
streamlit run streamlit/app.py
L'application est accessible à : http://localhost:8501

3. Application en ligne (Streamlit Cloud)
🔗 https://rentpredictionproject.streamlit.app

📊 Résultats des modèles
Modèle	MAE (BIF)	RMSE (BIF)	R²
Dummy (Moyenne)	125 230	160 450	0.000
Dummy (Médiane)	120 100	158 200	-0.012
Linear Regression	81 200	109 500	0.765
Ridge (α=1.0)	80 500	108 200	0.772
Lasso (α=0.1)	81 000	109 000	0.768
Decision Tree (depth=10)	85 000	115 000	0.740
Random Forest (n=100)	64 800	91 800	0.835
Gradient Boosting (n=300)	64 200	90 500	0.842
🏆 Modèle final sélectionné
Gradient Boosting Regressor

Paramètre	Valeur
learning_rate	0.05
n_estimators	300
max_depth	5
min_samples_split	2
random_state	42
Justification :

Meilleur R² (0.842) et plus faible MAE (64 200 BIF).

Bon compromis entre précision et temps d'inférence.

Robuste face aux outliers grâce à la nature ensembliste.

💼 Résultats & Valeur Business
L'application Streamlit permet désormais aux agents de MEDIABOX et aux propriétaires de :

✅ Saisir les caractéristiques d'un bien en moins de 1 minute.

✅ Obtenir une estimation instantanée, transparente et standardisée.

✅ Anticiper les fluctuations du marché grâce à des données historiques.

Le pipeline est totalement reproductible, et toutes les décisions d'ingénierie sont strictement justifiées par les observations de l'EDA.

📝 Auteurs
Équipe MEDIABOX Burundi

[Nom de l'étudiant]

[Nom de l'encadrant]
