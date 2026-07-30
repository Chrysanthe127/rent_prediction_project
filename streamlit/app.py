import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ================== CONFIGURATION ==================
st.set_page_config(
    page_title="Prédiction de loyer - Bujumbura",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Estimation du loyer mensuel")
st.markdown("Saisissez les caractéristiques de la maison pour obtenir une estimation en **BIF**.")

# ================== CONSTANTES ==================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model_tuned.pkl"
DATA_PATH = BASE_DIR / "data" / "feature_engineering_dataset.csv"

# ================== CHARGEMENT ==================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Modèle introuvable : {MODEL_PATH}")
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

@st.cache_data
def load_feature_order():
    if not DATA_PATH.exists():
        st.error(f"Fichier de données introuvable : {DATA_PATH}")
        st.stop()
    try:
        df = pd.read_csv(DATA_PATH)
        target = "LoyerMensuel_BIF"
        return [col for col in df.columns if col != target]
    except Exception as e:
        st.error(f"Erreur lors du chargement du CSV : {e}")
        st.stop()

model = load_model()
csv_cols = load_feature_order()

# Utiliser l'ordre des colonnes du modèle (plus fiable)
if hasattr(model, "feature_names_in_"):
    final_cols = list(model.feature_names_in_)
    # Vérifier que toutes les colonnes attendues sont présentes dans le CSV (au cas où)
    missing_in_csv = set(final_cols) - set(csv_cols)
    if missing_in_csv:
        st.warning(f"Colonnes du modèle non trouvées dans le CSV : {missing_in_csv}")
else:
    final_cols = csv_cols



# ================== LISTE DES QUARTIERS ==================
QUARTIERS = [
    "Buyenzi", "Bwiza", "Cibitoke", "Gasekebuye", "Gihosha", "Jabe",
    "Kamenge", "Kinama", "Kinanira", "Kiriri", "Musaga", "Ngagara",
    "Nyakabiga", "Rohero"
]

# ================== FORMULAIRE ==================
with st.form("prediction_form"):
    st.subheader("Caractéristiques de la maison")
    col1, col2 = st.columns(2)
    with col1:
        chambres = st.number_input("Nombre de chambres", min_value=1,  value=3, step=1)
        superficie = st.number_input("Superficie (m²)", min_value=10,  value=150, step=5)
        distance = st.number_input("Distance à la route principale (m)", min_value=0, value=200, step=10)
        age = st.number_input("Âge de la maison (années)", min_value=0, value=20, step=1)
    with col2:
        salon = st.selectbox("Salon", ["Non", "Oui"])
        sdb = st.selectbox("Salle de bain intérieure", ["Non", "Oui"])
        parking = st.selectbox("Parking", ["Non", "Oui"])
        meuble = st.selectbox("Meublé", ["Non", "Oui"])
        jardin = st.selectbox("Jardin", ["Non", "Oui"])
    quartier = st.selectbox("Quartier", QUARTIERS)
    submitted = st.form_submit_button("Estimer le loyer")

# ================== FONCTION DE PRÉDICTION ==================
def predict(chambres, superficie, distance, age,
            salon_val, sdb_val, parking_val, meuble_val, jardin_val,
            quartier):
    # --- Features dérivées ---
    superficie_ok = max(superficie, 1e-6)
    chambres_ok = max(chambres, 1)
    surface_par_chambre = superficie_ok / chambres_ok
    chambres_par_superficie = chambres / superficie_ok
    distance_log = np.log(distance + 1)
    age_log = np.log(age + 1)
    age_surface = age * superficie
    confort_score = salon_val + sdb_val + parking_val + meuble_val + jardin_val
    confort_surface = confort_score * superficie
    confort_chambres = confort_score * chambres

    # --- One-hot encoding du quartier ---
    quartier_dict = {f'nom__Quartier_{q}': 0 for q in QUARTIERS}
    quartier_dict[f'nom__Quartier_{quartier}'] = 1

    # --- Construction du dictionnaire de données ---
    data = {
        'num__Chambres': chambres,
        'num__Superficie_m2': superficie,
        'num__DistanceRoute_m': distance,
        'num__AgeMaison': age,
        'binary__Salon': salon_val,
        'binary__SalleDeBainInterieure': sdb_val,
        'binary__Parking': parking_val,
        'binary__Meuble': meuble_val,
        'binary__Jardin': jardin_val,
        **quartier_dict,
        'SurfaceParChambre': surface_par_chambre,
        'ChambresParSuperficie': chambres_par_superficie,
        'DistanceLog': distance_log,
        'AgeMaisonLog': age_log,
        'Age_Surface': age_surface,
        'Confort_Score': confort_score,
        'Confort_Surface': confort_surface,
        'Confort_Chambres': confort_chambres
    }

    # --- DataFrame ---
    input_df = pd.DataFrame([data])

    # Vérification des colonnes manquantes
    missing = set(final_cols) - set(input_df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    # Réordonnancement selon le modèle
    input_df = input_df[final_cols]

    # Prédiction
    return model.predict(input_df)[0]

# ================== EXÉCUTION DE LA PRÉDICTION ==================
if submitted:
    try:
        prediction = predict(
            chambres, superficie, distance, age,
            1 if salon == "Oui" else 0,
            1 if sdb == "Oui" else 0,
            1 if parking == "Oui" else 0,
            1 if meuble == "Oui" else 0,
            1 if jardin == "Oui" else 0,
            quartier
        )
        st.success(f"💰 Loyer estimé : **{prediction:,.0f} BIF**")
        st.balloons()
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction : {e}")
        # Mode débogage (affiché uniquement si l'utilisateur le demande)
        if st.checkbox("🔧 Afficher les détails techniques"):
            st.write("**Colonnes attendues par le modèle :**", final_cols)
            if 'input_df' in locals():
                st.write("**Colonnes fournies :**", input_df.columns.tolist())