# Rapport de Feature Engineering 

**Projet** : Système de Prédiction de Loyer — Bujumbura  
**Auteur** : Chrysanthe Beni Joy  
**Date** : 31 juillet 2026  


---

## 1. Contexte et objectifs

La phase de **Feature Engineering** s'inscrit dans la continuité de l'Analyse Exploratoire des Données (EDA). Son objectif est de créer de nouvelles variables à partir des données brutes afin de :

- Rendre explicites des informations implicites identifiées lors de l'EDA.
- Améliorer le pouvoir prédictif des modèles de régression.
- Réduire la dimensionnalité et le bruit.
- Faciliter l'interprétation des résultats par les métiers.

L'EDA a mis en évidence plusieurs signaux forts :

| Observation | Implication |
|-------------|-------------|
| Corrélation élevée entre `Chambres` et `Superficie_m2` (r ≈ 0.78) | Risque de multicolinéarité ; besoin d'un ratio pour capter la densité des pièces. |
| Effet positif de chaque équipement (Salon, SDB intérieure, Parking, Meublé, Jardin) sur le loyer | Possibilité de synthétiser ces cinq variables en un score de confort unique. |
| Certains quartiers (ex. Gasekebuye, Jabe, Kanyosha) ont moins de 5 % des observations | Risque de sur‑apprentissage ; nécessité d'un regroupement des classes rares. |
| Distributions asymétriques pour `DistanceRoute_m` et `AgeMaison` | Intérêt d'une transformation logarithmique pour réduire l'influence des valeurs extrêmes. |
| Relations non linéaires potentielles entre superficie, âge et confort | Création de variables d'interaction pour capturer ces effets combinés. |

---

## 2. Variables créées

### 2.1 Score de confort (`Confort_Score`)

**Justification**

Les cinq équipements binaires sont tous positivement corrélés au loyer. Leur présence simultanée dans un logement est fréquente ; un score synthétique permet de résumer cet effet additif.

**Construction**

```text
Confort_Score = Salon + SalleDeBainInterieure + Parking + Meuble + Jardin
```

Valeur minimale : 0 (aucun équipement) — Valeur maximale : 5 (tous les équipements).

**Bénéfices attendus**

- Réduction de la dimensionnalité (5 variables → 1).
- Capture de l'effet combiné des équipements.
- Amélioration de l'interprétabilité.

**Impact observé**

Le score de confort figure parmi les trois caractéristiques les plus importantes dans le modèle Gradient Boosting final, confirmant sa pertinence pour expliquer la variation des loyers.

---

### 2.2 Regroupement des quartiers rares

**Justification**

L'EDA a révélé que certains quartiers disposent de très peu d'observations :

| Quartier | Effectif | Part du total |
|----------|----------|---------------|
| Gasekebuye | 18 | 3,5 % |
| Jabe | 22 | 4,3 % |
| Kanyosha | 12 | 2,4 % |
| Musaga | 15 | 2,9 % |

Ces classes rares augmentent le risque de sur‑apprentissage et dégradent la capacité de généralisation du modèle.

**Méthode**

Tous les quartiers représentant moins de **5 %** des observations ont été fusionnés dans une catégorie unique :

```text
Autre
```

Le nombre de colonnes quartier est ainsi passé de 14 à 9 (8 quartiers conservés + 1 catégorie `Autre`).

**Bénéfices attendus**

- Réduction du bruit et de la variance.
- Meilleure généralisation sur des quartiers peu représentés.
- Simplification du modèle.

**Impact observé**

Le regroupement améliore la stabilité du modèle en validation croisée, avec une réduction notable de la variance des performances.

---

### 2.3 Ratio Chambres / Superficie (`ChambresParSuperficie`)

**Justification**

La corrélation élevée entre `Chambres` et `Superficie_m2` indique une forte redondance. Le ratio `Chambres / Superficie` mesure la **densité de pièces** par mètre carré. Une valeur élevée caractérise un logement avec de nombreuses chambres pour une surface donnée, ce qui peut influencer le loyer au‑delà de la seule superficie.

**Construction**

```text
ChambresParSuperficie = Nombre de chambres / Superficie (m²)
```

**Bénéfices attendus**

- Capturer la densité des pièces.
- Réduire la multicolinéarité.
- Apporter une information complémentaire.

**Décision finale**

Cette variable n'a **pas été retenue** dans le pipeline final. En effet, le jeu de données `processed_dataset.csv` utilisé pour l'entraînement contenait des variables déjà normalisées (RobustScaler). Le calcul du ratio après normalisation aurait perdu son interprétation physique et sa cohérence avec les données réelles.

---

### 2.4 Transformations logarithmiques

**Justification**

Les distributions de `DistanceRoute_m` et `AgeMaison` présentent une asymétrie positive marquée, avec des valeurs extrêmes éloignées de la médiane. Une transformation logarithmique permet de :

- Réduire l'influence des valeurs aberrantes.
- Linéariser les relations avec la variable cible.

**Formules**

```text
DistanceLog = log(1 + DistanceRoute_m)
AgeMaisonLog = log(1 + AgeMaison)
```

**Bénéfices attendus**

- Meilleure adéquation avec les modèles linéaires.
- Stabilité accrue des coefficients.
- Réduction de l'effet des outliers.

**Impact observé**

Les transformations logarithmiques améliorent la relation linéaire entre ces variables et le loyer, et sont donc conservées.

---

### 2.5 Variables d'interaction

**Justification**

Certaines relations entre caractéristiques peuvent être non linéaires et ne pas être capturées par les variables seules. Des interactions ont été créées pour représenter ces effets combinés.

**Variables créées**

| Variable | Formule | Description |
|----------|---------|-------------|
| `Confort_Surface` | `Confort_Score × Superficie_m2` | Effet du confort modulé par la surface. |
| `Confort_Chambres` | `Confort_Score × Chambres` | Effet du confort modulé par le nombre de chambres. |
| `Age_Surface` | `AgeMaison × Superficie_m2` | Effet de l'ancienneté combiné à la surface. |

**Bénéfices attendus**

- Capturer des relations non linéaires.
- Améliorer la précision du modèle.
- Révéler des synergies entre variables.

**Impact observé**

Ces trois variables d'interaction sont conservées car elles apportent un supplément d'information significatif au modèle.

---

## 3. Synthèse des décisions

| Feature | Justification (issue de l'EDA) | Décision |
|---------|--------------------------------|----------|
| `Confort_Score` | Effet additif des équipements sur le loyer | ✅ Conservée |
| Regroupement des quartiers rares | Classes < 5 % d'effectifs → bruit, sur‑apprentissage | ✅ Conservée |
| `ChambresParSuperficie` | Multicolinéarité Chambres / Superficie | ❌ Non conservée (problème de scaling) |
| `DistanceLog` | Asymétrie de la distance | ✅ Conservée |
| `AgeMaisonLog` | Asymétrie de l'âge | ✅ Conservée |
| `Confort_Surface` | Interaction confort × superficie | ✅ Conservée |
| `Confort_Chambres` | Interaction confort × chambres | ✅ Conservée |
| `Age_Surface` | Interaction âge × superficie | ✅ Conservée |

---

## 4. Conclusion

La phase de **Feature Engineering** a permis de transformer les observations de l'EDA en variables pertinentes et interprétables. Les créations retenues — notamment le score de confort, les transformations logarithmiques et les interactions — apportent des informations complémentaires qui améliorent la représentation des logements et facilitent l'apprentissage des modèles.

Le regroupement des quartiers rares contribue également à réduire le bruit et à renforcer la capacité de généralisation.

**Jeu de données final** :

```text
data/feature_engineering_dataset.csv
```

Ce jeu servira de base pour les phases de modélisation (entraînement des modèles de régression) et d'optimisation hyperparamétrique.

---
  
Bujumbura, juillet 2026