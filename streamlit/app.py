import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ================== CONFIGURATION DE LA PAGE ==================
st.set_page_config(
    page_title="Prédiction de loyer - Bujumbura",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ================== CSS PERSONNALISÉ ULTRA-MODERNE DARK ==================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Arrière-plan global hyper sombre avec dégradé subtil */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0F172A 0%, #090D16 50%, #030712 100%);
        color: #E2E8F0;
    }

    /* En-tête principal */
    .hero-header {
        text-align: center;
        padding: 1.5rem 0 2.5rem 0;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
        letter-spacing: -0.8px;
    }
    .hero-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        max-width: 620px;
        margin: 0 auto;
        line-height: 1.5;
    }

    /* Conteneur principal */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 800px;
    }

    /* ================== FORMULAIRE DARK NEON ================== */
    div[data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 30px rgba(56, 189, 248, 0.05);
    }

    /* Titres dans le formulaire */
    div[data-testid="stForm"] h3 {
        color: #F1F5F9 !important;
        font-weight: 700 !important;
        font-size: 1.25rem !important;
        margin-bottom: 1.5rem !important;
        border-bottom: 1px solid rgba(56, 189, 248, 0.15);
        padding-bottom: 0.6rem;
    }

    /* Labels */
    div[data-testid="stForm"] label {
        color: #38BDF8 !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* Champs texte / nombre / select */
    .stNumberInput input, .stSelectbox div[role="button"] {
        background-color: #0B1120 !important;
        color: #F1F5F9 !important;
        border-radius: 14px !important;
        border: 1px solid rgba(56, 189, 248, 0.25) !important;
        font-weight: 500 !important;
        padding: 0.5rem 0.8rem !important;
    }

    .stNumberInput input:focus, .stSelectbox div[role="button"]:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.3) !important;
    }

    /* Boutons +/- du number input */
    .stNumberInput button {
        background-color: #0B1120 !important;
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
        border-radius: 10px !important;
    }

    .stNumberInput button:hover {
        background-color: #1E293B !important;
        color: #7DD3FC !important;
    }

    /* ================== BOUTON SANS BLANC ================== */
    div[data-testid="stFormSubmitButton"] button {
        width: 100%;
        background: linear-gradient(135deg, #0284C7 0%, #4F46E5 100%) !important;
        color: #E0F2FE !important; /* Bleu très clair au lieu de blanc */
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 16px !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.95rem 1.5rem !important;
        margin-top: 1.2rem !important;
        box-shadow: 0 10px 25px -5px rgba(2, 132, 199, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-3px);
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 100%) !important;
        color: #F0F9FF !important;
        box-shadow: 0 15px 35px -5px rgba(56, 189, 248, 0.6);
        border-color: #38BDF8 !important;
    }

    /* ================== CARTE DE RÉSULTAT DARK NEON ================== */
    .result-card {
        background: rgba(15, 23, 42, 0.85);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 24px;
        padding: 2.5rem;
        margin-top: 2rem;
        text-align: center;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.9), 0 0 40px rgba(56, 189, 248, 0.15);
        animation: fadeIn 0.4s ease-out;
    }

    .result-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 1.2rem;
    }

    .result-price {
        font-size: 3.6rem;
        font-weight: 800;
        color: #F1F5F9;
        letter-spacing: -1.5px;
        line-height: 1.1;
        margin-bottom: 1.2rem;
        text-shadow: 0 0 25px rgba(56, 189, 248, 0.4);
    }

    .result-features {
        display: flex;
        justify-content: center;
        gap: 0.85rem;
        flex-wrap: wrap;
    }

    .feature-chip {
        background: #0B1120;
        color: #94A3B8;
        border: 1px solid rgba(56, 189, 248, 0.2);
        font-weight: 600;
        font-size: 0.88rem;
        padding: 0.55rem 1.1rem;
        border-radius: 14px;
    }

    /* Footer */
    .footer {
        text-align: center;
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(56, 189, 248, 0.1);
        color: #475569;
        font-size: 0.85rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# ================== EN-TÊTE ==================
st.markdown("""
<div class="hero-header">
    <div class="hero-title">Estimation de Loyer Intelligente</div>
    <div class="hero-subtitle">Obtenez une évaluation précise et instantanée du loyer mensuel pour vos biens immobiliers à Bujumbura.</div>
</div>
""", unsafe_allow_html=True)

# ================== CONSTANTES & CHARGEMENT ==================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model_tuned.pkl"
DATA_PATH = BASE_DIR / "data" / "feature_engineering_dataset.csv"

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"⚠️ Modèle introuvable : {MODEL_PATH}")
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.stop()

@st.cache_data
def load_feature_order():
    if not DATA_PATH.exists():
        st.error(f"⚠️ Fichier de données introuvable : {DATA_PATH}")
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
else:
    final_cols = csv_cols

QUARTIERS = [
    "Buyenzi", "Bwiza", "Cibitoke", "Gasekebuye", "Gihosha", "Jabe",
    "Kamenge", "Kinama", "Kinanira", "Kiriri", "Musaga", "Ngagara",
    "Nyakabiga", "Rohero"
]

# ================== FORMULAIRE PRINCIPAL ==================
with st.form("prediction_form"):
    st.subheader("📍 Emplacement & Dimensions")
    
    col_q, col_s = st.columns([1, 1])
    with col_q:
        quartier = st.selectbox("Quartier", QUARTIERS)
    with col_s:
        superficie = st.number_input("Superficie totale (m²)", min_value=10, value=150, step=5)

    col1, col2 = st.columns(2)
    with col1:
        chambres = st.number_input("Nombre de chambres", min_value=1, value=3, step=1)
        distance = st.number_input("Distance à la route (m)", min_value=0, value=200, step=10)
    with col2:
        age = st.number_input("Âge du bâtiment (années)", min_value=0, value=10, step=1)
        
    st.subheader("✨ Équipements & Confort")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        salon = st.selectbox("Salon principal", ["Non", "Oui"])
        sdb = st.selectbox("SDB Intérieure", ["Non", "Oui"])
    with c2:
        parking = st.selectbox("Espace Parking", ["Non", "Oui"])
        jardin = st.selectbox("Jardin privatif", ["Non", "Oui"])
    with c3:
        meuble = st.selectbox("Meublé", ["Non", "Oui"])

    submitted = st.form_submit_button("Calculer l'estimation du loyer")

# ================== MOTEUR DE PRÉDICTION ==================
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

# ================== RÉSULTATS ==================
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
        
        # Formatage du prix
        formatted_price = f"{prediction:,.0f}".replace(",", " ")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-badge">Résultat de l'estimation</div>
            <div class="result-price">{formatted_price} <span style="font-size: 1.5rem; font-weight: 600; color: #38BDF8;">BIF / mois</span></div>
            <div class="result-features">
                <span class="feature-chip">📍 {quartier}</span>
                <span class="feature-chip">📐 {superficie:.0f} m²</span>
                <span class="feature-chip">🛏️ {chambres} ch.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
        
    except Exception as e:
        st.error(f"Une erreur est survenue lors du calcul : {e}")
        with st.expander("Voir les détails d'erreur"):
            st.write(e)

# ================== PIED DE PAGE ==================
st.markdown("""
<div class="footer">
    © 2026 <strong>MEDIABOX Burundi</strong> · Solution d'évaluation immobilière par Machine Learning
</div>
""", unsafe_allow_html=True)
