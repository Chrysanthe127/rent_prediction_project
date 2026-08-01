import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ================== CONFIGURATION ==================
st.set_page_config(
    page_title="Prédiction de loyer - Bujumbura",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ================== CSS PERSONNALISÉ (design sombre, texte noir) ==================
st.markdown("""
<style>
    /* ====== POLICE ====== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * {
        font-family: 'Inter', sans-serif;
        color: #0F172A !important;  /* Texte noir forcé */
    }

    /* ====== FOND ====== */
    .stApp {
        background: #F1F5F9;  /* Fond gris clair */
    }

    /* ====== CONTENEUR ====== */
    .block-container {
        padding: 2rem 1.5rem;
        max-width: 800px;
        margin: 0 auto;
    }

    /* ====== TITRES ====== */
    h1 {
        text-align: center;
        font-weight: 800;
        font-size: 2.5rem;
        color: #0F172A !important;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .subtitle {
        text-align: center;
        color: #334155 !important;
        margin-top: -0.3rem;
        font-weight: 400;
        font-size: 1rem;
    }

    h2, h3, .stSubheader {
        color: #0F172A !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }

    /* ====== FORMULAIRE (carte blanche avec ombre) ====== */
    div[data-testid="stForm"] {
        background: #FFFFFF;
        padding: 2rem 2rem 1.8rem;
        border-radius: 24px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 10px 25px -8px rgba(0,0,0,0.08);
        transition: box-shadow 0.3s ease;
    }

    div[data-testid="stForm"]:hover {
        box-shadow: 0 20px 40px -12px rgba(0,0,0,0.12);
    }

    /* ====== LABELS ====== */
    .stNumberInput > label,
    .stSelectbox > label,
    .stCheckbox label {
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-bottom: 0.25rem !important;
    }

    /* ====== INPUTS & SELECT ====== */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: #F8FAFC !important;
        color: #0F172A !important;
        border-radius: 12px !important;
        border: 1px solid #CBD5E1 !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }

    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus-within {
        border-color: #0EA5E9 !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15), inset 0 2px 4px rgba(0,0,0,0.02) !important;
        background: #FFFFFF !important;
    }

    /* ====== CHECKBOX (équipements) ====== */
    .stCheckbox {
        margin: 0.4rem 0;
    }
    .stCheckbox label {
        display: flex;
        align-items: center;
        color: #0F172A !important;
        font-weight: 500 !important;
        background: #F8FAFC;
        padding: 0.5rem 0.8rem;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        transition: all 0.2s ease;
        width: 100%;
        cursor: pointer;
        text-transform: none;
        font-size: 0.9rem !important;
    }
    .stCheckbox label:hover {
        background: #F1F5F9;
        border-color: #94A3B8;
    }
    .stCheckbox div[data-testid="stCheckbox"] {
        border-radius: 8px;
        margin-right: 8px;
    }

    /* ====== BOUTON PRINCIPAL ====== */
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        background: #0F172A !important;
        color: #FFFFFF !important;
        border: none;
        border-radius: 16px;
        font-size: 1.1rem;
        font-weight: 700;
        padding: 0.8rem 1.5rem;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.25);
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
        cursor: pointer;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background: #1E293B !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.35);
    }

    div[data-testid="stFormSubmitButton"] button:active {
        transform: translateY(0px);
        box-shadow: 0 4px 10px rgba(15, 23, 42, 0.2);
    }

    /* ====== MESSAGES SUCCÈS ====== */
    div[data-testid="stSuccess"] {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        border-radius: 18px;
        padding: 1.5rem;
        color: #0F172A !important;
        font-weight: 500;
        text-align: center;
    }
    div[data-testid="stSuccess"] strong {
        color: #0F172A;
    }

    /* ====== MESSAGES ERREUR ====== */
    div[data-testid="stAlert"] {
        background: #FEF2F2;
        border: 1px solid #FCA5A5;
        border-radius: 18px;
        padding: 1.2rem;
        color: #0F172A !important;
        font-weight: 500;
    }

    /* ====== WARNING ====== */
    div[data-testid="stWarning"] {
        background: #FFFBEB;
        border: 1px solid #FDE68A;
        border-radius: 18px;
        padding: 1rem;
        color: #0F172A !important;
    }

    /* ====== CARTE RÉSULTAT ====== */
    .result-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 24px;
        padding: 1.8rem;
        margin-top: 1.5rem;
        box-shadow: 0 8px 20px -8px rgba(0,0,0,0.06);
        text-align: center;
        transition: box-shadow 0.3s ease;
    }
    .result-card:hover {
        box-shadow: 0 12px 30px -8px rgba(0,0,0,0.10);
    }
    .result-card .label {
        color: #334155 !important;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .result-card .price {
        font-size: 3.2rem;
        font-weight: 800;
        color: #0F172A !important;
        margin: 0.2rem 0 0.3rem;
        line-height: 1.2;
    }
    .result-card .details {
        color: #334155 !important;
        font-size: 0.9rem;
    }

    /* ====== PIED DE PAGE ====== */
    .footer {
        text-align: center;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #E2E8F0;
        color: #64748B !important;
        font-size: 0.8rem;
        letter-spacing: 0.3px;
    }

    /* ====== SCROLLBAR ====== */
    ::-webkit-scrollbar {
        width: 6px;
        background: #F1F5F9;
    }
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }

    /* ====== RESPONSIVE ====== */
    @media (max-width: 640px) {
        h1 { font-size: 1.8rem; }
        div[data-testid="stForm"] { padding: 1.2rem; }
        .result-card .price { font-size: 2.4rem; }
    }
</style>
""", unsafe_allow_html=True)

# ================== TITRE ==================
st.title("Estimation du loyer mensuel")
st.markdown('<p class="subtitle">Saisissez les caractéristiques de la maison pour obtenir une estimation en <strong>BIF</strong>.</p>', unsafe_allow_html=True)

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

if hasattr(model, "feature_names_in_"):
    final_cols = list(model.feature_names_in_)
    missing_in_csv = set(final_cols) - set(csv_cols)
    if missing_in_csv:
        st.warning(f"Colonnes du modèle non trouvées dans le CSV : {missing_in_csv}")
else:
    final_cols = csv_cols

# ================== QUARTIERS ==================
QUARTIERS = [
    "Buyenzi", "Bwiza", "Cibitoke", "Gasekebuye", "Gihosha", "Jabe",
    "Kamenge", "Kinama", "Kinanira", "Kiriri", "Musaga", "Ngagara",
    "Nyakabiga", "Rohero"
]

# ================== FORMULAIRE ==================
with st.form("prediction_form"):
    st.subheader("Caractéristiques du logement")
    col1, col2 = st.columns(2)
    with col1:
        chambres = st.number_input("Nombre de chambres", min_value=1, value=3, step=1)
        superficie = st.number_input("Superficie (m²)", min_value=10, value=150, step=5)
        distance = st.number_input("Distance route principale (m)", min_value=0, value=200, step=10)
        age = st.number_input("Âge de la maison (années)", min_value=0, value=20, step=1)
    with col2:
        salon = st.selectbox("Salon", ["Non", "Oui"])
        sdb = st.selectbox("Salle de bain intérieure", ["Non", "Oui"])
        parking = st.selectbox("Parking", ["Non", "Oui"])
        meuble = st.selectbox("Meublé", ["Non", "Oui"])
        jardin = st.selectbox("Jardin", ["Non", "Oui"])
    quartier = st.selectbox("Quartier", QUARTIERS)
    submitted = st.form_submit_button("Estimer le loyer maintenant")

# ================== PRÉDICTION ==================
def predict(chambres, superficie, distance, age,
            salon_val, sdb_val, parking_val, meuble_val, jardin_val,
            quartier):
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

    quartier_dict = {f'nom__Quartier_{q}': 0 for q in QUARTIERS}
    quartier_dict[f'nom__Quartier_{quartier}'] = 1

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

    input_df = pd.DataFrame([data])
    missing = set(final_cols) - set(input_df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    input_df = input_df[final_cols]
    return model.predict(input_df)[0]

# ================== AFFICHAGE RÉSULTAT ==================
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
        st.markdown(f"""
        <div class="result-card">
            <div class="label">Loyer mensuel estimé</div>
            <div class="price">{prediction:,.0f} BIF</div>
            <div class="details">{superficie:.0f} m² · {chambres} chambre(s) · {quartier}</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        if st.checkbox("Afficher les détails techniques"):
            st.write("**Colonnes attendues par le modèle :**", final_cols)
            if 'input_df' in locals():
                st.write("**Colonnes fournies :**", input_df.columns.tolist())

# ================== PIED DE PAGE ==================
st.markdown("""
<div class="footer">
    © 2026 MEDIABOX Burundi · Estimateur intelligent de loyers
</div>
""", unsafe_allow_html=True)