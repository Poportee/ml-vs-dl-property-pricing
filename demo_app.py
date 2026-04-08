import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import torch
import joblib
import requests
import math
import os
import sys
import warnings
from datetime import datetime

# Permet d'importer l'architecture du modèle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from ft_transformer_net_model import FTTransformer

warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Prédiction Valeur Foncière", page_icon="🏠", layout="centered")

# --- VARIABLES GLOBALES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
STATS_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'cp_stats.parquet')
COMMUNES_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'communes-france-2025.csv')


# ==================================================================================
# MISE EN CACHE DES DONNÉES ET MODÈLES (Crucial pour la vitesse web)
# ==================================================================================

@st.cache_data
def load_global_stats():
    """Charge les stats historiques en cache."""
    if os.path.exists(STATS_FILE):
        try:
            return pd.read_parquet(STATS_FILE).set_index('code_postal')
        except Exception:
            pass
    return pd.DataFrame()

@st.cache_data
def load_commune_data():
    """Charge les données des communes en cache."""
    if not os.path.exists(COMMUNES_FILE):
        return None
    return pd.read_csv(COMMUNES_FILE, usecols=['code_insee','latitude_centre','longitude_centre','densite','altitude_moyenne','population'], dtype={'code_insee': str})

@st.cache_resource
def load_models():
    """Charge les modèles ML/DL en mémoire une seule fois."""
    try:
        cat_encoder = joblib.load(os.path.join(MODELS_DIR, 'encoder_categorielles_lgbm.pkl'))
        lgb_model = joblib.load(os.path.join(MODELS_DIR, 'modele_lightgbm_finalRMSE.pkl'))
        num_scaler = joblib.load(os.path.join(MODELS_DIR, 'num_scaler.pkl'))
        
        # FT-Transformer
        model_ft = FTTransformer(
            n_num=25, cat_dims=[5, 5869, 98, 16592, 13], d_emb=128, 
            n_layers=3, n_heads=8, ffn_factor=4/3, attn_dropout=0.2, 
            ffn_dropout=0.1, resid_dropout=0.0
        )
        model_ft.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'ft_transformer_best_model_weights.pt'), map_location='cpu'))
        model_ft.eval()
        
        return cat_encoder, lgb_model, num_scaler, model_ft
    except Exception as e:
        st.error(f"Erreur de chargement des modèles : {e}")
        return None, None, None, None


# ==================================================================================
# FONCTIONS UTILITAIRES (Identiques à votre script)
# ==================================================================================

def get_historical_stats(code_postal, global_stats_df):
    if not global_stats_df.empty and code_postal in global_stats_df.index:
        return global_stats_df.loc[code_postal].to_dict()

    # Fallback
    dept = str(code_postal)[:2]
    mock_prices_m2 = {
        '75': 10500, '92': 7500, '94': 5500, '93': 4500, '69': 5000, 
        '33': 4800, '13': 4000, '06': 5500, '31': 3800, '44': 4000, '34': 3500, '59': 3000
    }
    base_price = mock_prices_m2.get(dept, 2200)

    return {
        'prix_m2_median_cp': base_price, 'prix_m2_mean_cp': base_price, 'prix_m2_std_cp': base_price * 0.2,
        'surface_m2_median_cp': 70, 'nb_pieces_mean_cp': 3, 'transactions_cp': 150,
        'prix_m2_median_cp_last_12m': base_price * 1.02, 'transactions_cp_last_12m': 30
    }

def haversine(coord1, coord2):
    R = 6371
    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])
    a = math.sin((lat2 - lat1) / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def get_geo_data(address):
    url = "https://api-adresse.data.gouv.fr/search/"
    try:
        r = requests.get(url, params={'q': address, 'limit': 1}).json()
        if r['features']:
            props = r['features'][0]['properties']
            coords = r['features'][0]['geometry']['coordinates']
            return {
                'lat': coords[1], 'lon': coords[0],
                'code_insee': props.get('citycode', '00000'),
                'code_postal': props.get('postcode', '00000'),
                'code_voie': props.get('id', '0000')[5:9] if len(props.get('id', '')) > 9 else '0000',
                'city': props.get('city', '')
            }
    except Exception:
        pass
    return None

def safe_remap(val, max_dim):
    if val == -1: return 0 
    new_val = val + 1
    if new_val >= max_dim: return 0 
    return new_val


# ==================================================================================
# INTERFACE UTILISATEUR STREAMLIT
# ==================================================================================

st.title("🏠 Estimateur de Valeur Foncière (ML vs DL)")
st.markdown("Ce projet compare les performances d'un modèle **LightGBM** (Machine Learning) et d'un **FT-Transformer** (Deep Learning) pour l'estimation immobilière en France.")

# --- Chargement silencieux en arrière-plan ---
global_stats_df = load_global_stats()
communes_df = load_commune_data()
cat_encoder, lgb_model, num_scaler, model_ft = load_models()

# --- Formulaire de saisie ---
with st.form("prediction_form"):
    st.subheader("Informations du bien")
    
    address_input = st.text_input("Adresse complète", placeholder="Ex: 10 rue de la paix 75002 Paris")
    
    col1, col2 = st.columns(2)
    with col1:
        type_local = st.selectbox("Type de bien", ["Appartement", "Maison"])
        surface = st.number_input("Surface réelle bâtie (m²)", min_value=9.0, value=50.0, step=1.0)
    with col2:
        nb_pieces = st.number_input("Nombre de pièces principales", min_value=1, value=2, step=1)
        
    submit_button = st.form_submit_button("Estimer le prix")

# --- Logique de prédiction ---
if submit_button:
    if not address_input:
        st.warning("Veuillez saisir une adresse.")
    else:
        with st.spinner("Recherche des coordonnées géographiques..."):
            geo_info = get_geo_data(address_input)
            
        if not geo_info:
            st.error("Adresse introuvable via l'API Data Gouv. Essayez d'être plus précis.")
        else:
            st.success(f"📍 Localisé : {geo_info['city']} ({geo_info['code_postal']}) - INSEE: {geo_info['code_insee']}")
            
            with st.spinner("Calcul des features et Inférence..."):
                # --- Création du vecteur ---
                commune_info = communes_df[communes_df['code_insee'] == geo_info['code_insee']] if communes_df is not None else pd.DataFrame()
                
                densite = float(commune_info['densite'].values[0]) if not commune_info.empty else 100
                pop = float(commune_info['population'].values[0]) if not commune_info.empty else 1000
                alt = float(commune_info['altitude_moyenne'].values[0]) if not commune_info.empty else 100
                lat_centre = float(commune_info['latitude_centre'].values[0]) if not commune_info.empty else geo_info['lat']
                lon_centre = float(commune_info['longitude_centre'].values[0]) if not commune_info.empty else geo_info['lon']

                dist_centre = haversine((geo_info['lat'], geo_info['lon']), (lat_centre, lon_centre))
                stats = get_historical_stats(geo_info['code_postal'], global_stats_df)
                
                df_input = pd.DataFrame([{
                    'type_local': type_local, 'surface_reelle_bati': surface, 'nb_pieces_principales': nb_pieces,
                    'code_postal': geo_info['code_postal'], 'code_departement': geo_info['code_postal'][:2], 
                    'code_voie': geo_info['code_voie'], 'nombre_de_lots': 1.0, 'surface_m2': surface, 
                    'source_year': datetime.now().year, 'code_insee': int(geo_info['code_insee']) if geo_info['code_insee'].isdigit() else 0,
                    'lat': geo_info['lat'], 'lon': geo_info['lon'], 'distance_centre_ville': dist_centre,
                    'densite_commune': densite, 'altitude_moyenne_commune': alt, 'population_commune': pop,
                    'prix_m2_median_cp': stats['prix_m2_median_cp'], 'prix_m2_mean_cp': stats['prix_m2_mean_cp'], 
                    'prix_m2_std_cp': stats['prix_m2_std_cp'], 'surface_m2_median_cp': stats['surface_m2_median_cp'], 
                    'nb_pieces_mean_cp': stats['nb_pieces_mean_cp'], 'transactions_cp': stats['transactions_cp'],
                    'prix_m2_median_cp_last_12m': stats['prix_m2_median_cp_last_12m'], 'transactions_cp_last_12m': stats['transactions_cp_last_12m'],
                    'densite_x_population': densite * pop, 'densite_population_ratio': pop / (densite if densite > 0 else 1),
                    'densite_altitude_ratio': densite / (alt if alt > 0 else 1),
                    'attractivite_simple': pop / (dist_centre if dist_centre > 0 else 1),
                    'distance_x_densite': dist_centre * densite, 'month_mutation': datetime.now().month
                }])

                FEATURES_ORDER = [
                    'type_local', 'surface_reelle_bati', 'nb_pieces_principales', 'code_postal', 'code_departement', 'code_voie',
                    'nombre_de_lots', 'surface_m2', 'source_year', 'code_insee', 'lat', 'lon', 'distance_centre_ville', 
                    'densite_commune', 'altitude_moyenne_commune', 'population_commune','prix_m2_median_cp','prix_m2_mean_cp',
                    'prix_m2_std_cp', 'surface_m2_median_cp','nb_pieces_mean_cp','transactions_cp','prix_m2_median_cp_last_12m',
                    'transactions_cp_last_12m','densite_x_population','densite_population_ratio','densite_altitude_ratio',
                    'attractivite_simple','distance_x_densite', 'month_mutation'
                ]

                # --- Inférence ML / DL ---
                try:
                    # LightGBM
                    X_lgb = df_input[FEATURES_ORDER].copy()
                    cat_cols = ['type_local', 'code_postal', 'code_departement', 'code_voie']
                    X_lgb[cat_cols] = cat_encoder.transform(X_lgb[cat_cols])
                    pred_lgb_log = lgb_model.predict(X_lgb)[0]
                    prix_lgbm = np.exp(pred_lgb_log)

                    # FT-Transformer
                    X_trans = df_input[FEATURES_ORDER].copy()
                    CATEGORICAL_COLS = ['type_local', 'code_postal', 'code_departement', 'code_voie', 'month_mutation']
                    NUMERICAL_COLS = [c for c in FEATURES_ORDER if c not in CATEGORICAL_COLS]

                    X_trans[cat_cols] = cat_encoder.transform(X_trans[cat_cols]).astype(int)
                    limits = {'type_local': 5, 'code_postal': 5869, 'code_departement': 98, 'code_voie': 16592, 'month_mutation': 13}
                    for col in cat_cols:
                        X_trans[col] = X_trans[col].apply(lambda x: safe_remap(x, limits[col]))
                    X_trans['month_mutation'] = safe_remap(int(X_trans['month_mutation'].values[0]) - 1, limits['month_mutation'])
                    
                    X_trans[NUMERICAL_COLS] = num_scaler.transform(X_trans[NUMERICAL_COLS])

                    x_num_tensor = torch.tensor(X_trans[NUMERICAL_COLS].values, dtype=torch.float32)
                    x_cat_tensor = torch.tensor(X_trans[CATEGORICAL_COLS].values, dtype=torch.long)
                    
                    with torch.no_grad():
                        pred_log_trans = model_ft(x_num_tensor, x_cat_tensor).item()
                    
                    TARGET_MEAN, TARGET_STD = 12.7120, 0.7775
                    prix_ft = np.exp((pred_log_trans * TARGET_STD) + TARGET_MEAN)

                    # --- Affichage des résultats ---
                    st.markdown("---")
                    st.subheader("💰 Résultats de l'Estimation")
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric(label="Modèle LightGBM (ML)", value=f"{prix_lgbm:,.0f} €".replace(',', ' '))
                    with res_col2:
                        st.metric(label="FT-Transformer (Deep Learning)", value=f"{prix_ft:,.0f} €".replace(',', ' '))

                except Exception as e:
                    st.error(f"Une erreur est survenue lors de la prédiction : {e}")