# IOGA - Prédiction Hybride de Séismes par Intelligence Artificielle

Ce projet repose sur une **approche hybride** innovante : il combine des modèles physiques traditionnels avec le Machine Learning (XGBoost) pour affiner la localisation et la magnitude des séismes à partir des données **DYFI** (Did You Feel It?).

## 🚀 Fonctionnement de l'IA Hybride

### 1. La Base Physique
Le système ne part pas de zéro. Il utilise d'abord des principes géophysiques pour établir une première estimation :
- **Estimation d'épicentre :** Calcul par poids CDI (Community Decimal Intensity).
- **Prédiction de Magnitude :** Utilisation de la formule physique issue des travaux de **W. H. Bakun et C. M. Wentworth** pour une première évaluation basée sur l'atténuation de l'intensité.
- **Formules de distance :** Utilisation de la distance Haversine pour le calcul spatial.

### 2. La Correction par l'IA
L'IA intervient pour corriger les biais systématiques des formules physiques. En analysant l'écart entre les prédictions physiques et les données réelles, le modèle **XGBoost** apprend à compenser les erreurs de localisation et de magnitude.

## 📊 Performances et Limites

D'après les derniers tests d'entraînement (basés sur **2171 événements**) :

| Métrique | Performance (Test) | Amélioration vs Physique |
| :--- | :--- | :--- |
| **Magnitude (MAE)** | **0.4050** | **+51.8%** |
| **Localisation (Erreur Moyenne)** | **88.68 km** | **+21.4%** |

**Note sur la précision :** Le modèle affiche une solide amélioration par rapport aux méthodes classiques. Cependant, il n'est pas encore parfait en raison de la taille limitée du dataset d'entraînement. Une augmentation du nombre d'événements permettrait de réduire davantage l'écart entre l'entraînement (67 km) et le test (88 km) qui est actuellement distant de 21 km.

## 🛠️ Installation

1. **Cloner le projet** :
   ```bash
   git clone https://github.com/votre-username/IOGA_train.git
   cd IOGA_train
   ```

2. **Créer un environnement virtuel** :
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate | Mac/Linux: source venv/bin/activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Utilisation
```bash
python train.py
```
Les modèles sont sauvegardés dans le dossier `models/`.

## 👤 Auteur
**Nalina RAN**
