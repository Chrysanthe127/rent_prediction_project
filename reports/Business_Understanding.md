# Business Understanding – Phase 1

**Projet** : Système de Prédiction de Loyer — Bujumbura  
**Auteur** : Équipe MEDIABOX Burundi  
**Date** : 31 juillet 2026  
**Version** : 1.0  
**Statut** : Final

---

## 1. Problème métier

Le marché immobilier résidentiel à Bujumbura présente une grande variation des loyers selon plusieurs caractéristiques des logements, telles que :

- le nombre de chambres ;
- la superficie ;
- la présence d'un parking ;
- la présence d'un jardin ;
- l'état d'ameublement ;
- l'âge de la maison ;
- le quartier.

Les propriétaires, les agences immobilières et les futurs locataires ont souvent des difficultés à estimer un loyer juste et cohérent. Une estimation réalisée uniquement de manière subjective peut conduire à une sous‑évaluation ou à une surestimation du prix.

**Objectif métier** : développer un modèle d'intelligence artificielle capable d'estimer automatiquement le loyer mensuel d'une maison à partir de ses caractéristiques physiques et géographiques.

---

## 2. Objectif du projet

Construire un modèle de **Machine Learning** permettant de prédire le `LoyerMensuel_BIF` d'une maison située à Bujumbura en utilisant les informations disponibles dans le jeu de données.

Le système pourra ensuite être intégré dans une application Web ou Mobile afin de fournir une estimation rapide et fiable du loyer.

---

## 3. Variable cible

| Élément | Détail |
|---------|--------|
| **Nom** | `LoyerMensuel_BIF` |
| **Type** | Numérique (entier) |
| **Unité** | Francs Burundais (BIF) |
| **Rôle** | Variable à prédire |

Cette variable représente le montant du loyer mensuel d'une maison exprimé en Francs Burundais (BIF).

---

## 4. Type de problème

Il s'agit d'un problème de **Régression supervisée**.

**Pourquoi ?**

- La variable à prédire est une valeur numérique continue.
- Les données historiques contiennent déjà la valeur réelle du loyer.
- Le modèle apprend la relation entre les caractéristiques de la maison et son loyer.

---

## 5. Données disponibles

Le jeu de données contient **510 observations** et **12 variables**.

### Description des variables

| Colonne | Type | Description |
|---------|------|-------------|
| `IdentifiantMaison` | Entier | Identifiant unique — à exclure du modèle |
| `Chambres` | Flottant | Nombre de chambres |
| `Salon` | Catégoriel (Oui/Non) | Présence d'un salon |
| `SalleDeBainInterieure` | Catégoriel (Oui/Non) | Salle de bain intérieure |
| `Parking` | Catégoriel (Oui/Non) | Présence d'un parking |
| `Meuble` | Catégoriel (Oui/Non) | Maison meublée ou non |
| `Jardin` | Catégoriel (Oui/Non) | Présence d'un jardin |
| `Superficie_m2` | Flottant | Surface habitable en m² |
| `DistanceRoute_m` | Flottant | Distance à la route principale (m) |
| `Quartier` | Catégoriel (nominal) | Quartier de Bujumbura |
| `AgeMaison` | Flottant | Âge de la maison (années) |
| `LoyerMensuel_BIF` | Entier | Variable cible (BIF) |

> **Remarque** : la variable `IdentifiantMaison` sert uniquement d'identifiant unique et ne sera pas utilisée comme variable explicative lors de l'entraînement.

---

## 6. Contraintes du projet

Plusieurs contraintes doivent être prises en compte avant la modélisation :

### a) Données manquantes

Le jeu de données contient **25 valeurs manquantes** (soit environ 4,9 % du total) dans la majorité des variables explicatives. Ces valeurs devront être traitées lors du prétraitement (suppression ou imputation).

### b) Taille du jeu de données

Le dataset contient **510 lignes**, ce qui constitue un volume relativement réduit. Il faudra donc limiter le risque de sur‑apprentissage et utiliser une validation appropriée (cross‑validation).

### c) Variable catégorielle "Quartier"

Le quartier influence fortement le prix du loyer. Cette variable étant catégorielle, elle devra être transformée (par exemple avec un encodage one‑hot) avant l'entraînement du modèle.

### d) Qualité des données

Certaines variables peuvent contenir des valeurs atypiques ou des distributions déséquilibrées. Une analyse exploratoire sera nécessaire afin de détecter les valeurs aberrantes et vérifier la cohérence des données.

---

## 7. Utilisateurs concernés

Le modèle pourra être utilisé par :

- les agences immobilières ;
- les propriétaires ;
- les locataires ;
- les plateformes immobilières ;
- les développeurs souhaitant intégrer une estimation automatique dans une application Web ou Mobile.

---

## 8. Usage final

À l'issue du projet, le modèle sera intégré dans une **API** (par exemple avec FastAPI ou Flask). Une application Web ou Mobile enverra les caractéristiques d'une maison à cette API, qui retournera une estimation du loyer mensuel.

Cette solution permettra d'obtenir une estimation :

- rapide ;
- cohérente ;
- fondée sur les données historiques plutôt que sur une simple appréciation subjective.

---

**Équipe MEDIABOX Burundi**  
Bujumbura, juillet 2026