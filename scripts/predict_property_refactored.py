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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ft_transformer_net_model import FTTransformer

warnings.filterwarnings("ignore", category=UserWarning)

GLOBAL_STATS_DF = None
STATS_FILE = "./data/created_data/cp_stats.parquet"

# ==================================================================================
# 1. FONCTIONS UTILITAIRES & API (AVEC FALLBACK PORTFOLIO)
# ==================================================================================

def load_global_stats():
    """Charge le fichier de statistiques précalculé en mémoire s'il existe."""
    global GLOBAL_STATS_DF
    if GLOBAL_STATS_DF is None:
        if os.path.exists(STATS_FILE):
            try:
                GLOBAL_STATS_DF = pd.read_parquet(STATS_FILE).set_index('code_postal')
                print(f"[Info] Fichier historique '{STATS_FILE}' chargé avec succès.")
            except Exception as e:
                print(f"[Avertissement] Impossible de lire {STATS_FILE} : {e}")
                GLOBAL_STATS_DF = pd.DataFrame()
        else:
            print(f"[Avertissement] Fichier '{STATS_FILE}' introuvable. Activation du mode Fallback (données simulées).")
            GLOBAL_STATS_DF = pd.DataFrame()

def get_historical_stats(code_postal):
    """
    Tente de récupérer les agrégats exacts depuis le dataset historique.
    En cas d'échec ou d'absence du code postal, bascule sur une simulation (Fallback).
    """
    # 1. Tenter de charger les vraies statistiques
    if GLOBAL_STATS_DF is None:
        load_global_stats()

    # 2. Si les données existent pour ce code postal, on les utilise !
    if not GLOBAL_STATS_DF.empty and code_postal in GLOBAL_STATS_DF.index:
        print(f"[Info] Statistiques réelles trouvées pour le CP {code_postal}.")
        return GLOBAL_STATS_DF.loc[code_postal].to_dict()

    # 3. FALLBACK : Simulation si fichier absent ou code postal inconnu
    print(f"[Info] Utilisation des statistiques simulées (Fallback) pour le CP {code_postal}.")
    dept = str(code_postal)[:2]
    
    # Heuristique réaliste pour la démonstration
    mock_prices_m2 = {
        '75': 10500, '92': 7500, '94': 5500, '93': 4500, # IDF proche
        '69': 5000, '33': 4800, '13': 4000, '06': 5500,  # Grandes métropoles
        '31': 3800, '44': 4000, '34': 3500, '59': 3000
    }
    
    base_price = mock_prices_m2.get(dept, 2200)

    return {
        'prix_m2_median_cp': base_price,
        'prix_m2_mean_cp': base_price,
        'prix_m2_std_cp': base_price * 0.2,
        'surface_m2_median_cp': 70,
        'nb_pieces_mean_cp': 3,
        'transactions_cp': 150,
        'prix_m2_median_cp_last_12m': base_price * 1.02,
        'transactions_cp_last_12m': 30
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

def load_commune_data():
    local_path = "./data/communes-france-2025.csv"
    if not os.path.exists(local_path):
        return None
    return pd.read_csv(local_path, usecols=['code_insee','latitude_centre','longitude_centre','densite','altitude_moyenne','population'], dtype={'code_insee': str})

# ==================================================================================
# 2. PROGRAMME PRINCIPAL
# ==================================================================================

def main():
    print("\n=== Outil de Prévision de Valeur Foncière ===")

    # --- Saisie Utilisateur ---
    address_input = input("Entrez l'adresse complète du bien (ex: 10 rue de la paix 75002 Paris): ")
    type_local = input("Type (Appartement / Maison) : ").capitalize()
    if type_local not in ['Appartement', 'Maison']: type_local = 'Appartement'
    
    try:
        surface = float(input("Surface réelle bâtie (m2) : "))
        nb_pieces = float(input("Nombre de pièces principales : "))
    except ValueError:
        print("Erreur: Veuillez entrer des nombres valides.")
        return
    nb_lots = 1.0 

    geo_info = get_geo_data(address_input)
    if not geo_info:
        print("Erreur: Adresse introuvable via l'API.")
        return
    print(f"-> Localisé à : {geo_info['city']} ({geo_info['code_postal']}) - INSEE: {geo_info['code_insee']}")
    
    # --- Création du vecteur d'entrée ---
    communes_df = load_commune_data()
    commune_info = communes_df[communes_df['code_insee'] == geo_info['code_insee']] if communes_df is not None else pd.DataFrame()

    densite = float(commune_info['densite'].values[0]) if not commune_info.empty else 100
    pop = float(commune_info['population'].values[0]) if not commune_info.empty else 1000
    alt = float(commune_info['altitude_moyenne'].values[0]) if not commune_info.empty else 100
    lat_centre = float(commune_info['latitude_centre'].values[0]) if not commune_info.empty else geo_info['lat']
    lon_centre = float(commune_info['longitude_centre'].values[0]) if not commune_info.empty else geo_info['lon']

    dist_centre = haversine((geo_info['lat'], geo_info['lon']), (lat_centre, lon_centre))
    stats = get_historical_stats(geo_info['code_postal'])
    
    df_input = pd.DataFrame([{
        'type_local': type_local, 'surface_reelle_bati': surface, 'nb_pieces_principales': nb_pieces,
        'code_postal': geo_info['code_postal'], 'code_departement': geo_info['code_postal'][:2], 
        'code_voie': geo_info['code_voie'], 'nombre_de_lots': nb_lots, 'surface_m2': surface, 
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

    print("\n--- INFERENCE ---")

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')

    # --- Estimation LightGBM ---
    try:
        # L'encodeur d'origine
        cat_encoder = joblib.load(os.path.join(MODELS_DIR, 'encoder_categorielles_lgbm.pkl'))
        lgb_model = joblib.load(os.path.join(MODELS_DIR, 'modele_lightgbm_finalRMSE.pkl')) 
        
        X_lgb = df_input[FEATURES_ORDER].copy()
        cat_cols = ['type_local', 'code_postal', 'code_departement', 'code_voie']
        X_lgb[cat_cols] = cat_encoder.transform(X_lgb[cat_cols])
        
        pred_lgb_log = lgb_model.predict(X_lgb)[0]
        print(f"Estimation LightGBM       : {np.exp(pred_lgb_log):,.0f} €")
    except Exception as e:
        print(f"Erreur LightGBM : {e}")

    # --- Estimation FT-Transformer ---
    try:
        num_scaler = joblib.load(os.path.join(MODELS_DIR, 'num_scaler.pkl'))
        
        X_trans = df_input[FEATURES_ORDER].copy()
        CATEGORICAL_COLS = ['type_local', 'code_postal', 'code_departement', 'code_voie', 'month_mutation']
        NUMERICAL_COLS = [c for c in FEATURES_ORDER if c not in CATEGORICAL_COLS]

        # 1. HACK: Utilisation de l'encodeur LightGBM pour les 4 premières colonnes du Transformer
        cat_cols_lgbm = ['type_local', 'code_postal', 'code_departement', 'code_voie']
        X_trans[cat_cols_lgbm] = cat_encoder.transform(X_trans[cat_cols_lgbm]).astype(int)
        
        limits = {'type_local': 5, 'code_postal': 5869, 'code_departement': 98, 'code_voie': 16592, 'month_mutation': 13}

        # 2. Remap des index (comme dans l'entraînement du DL)
        def safe_remap(val, max_dim):
            if val == -1: return 0 
            new_val = val + 1
            if new_val >= max_dim: return 0 
            return new_val

        for col in cat_cols_lgbm:
            X_trans[col] = X_trans[col].apply(lambda x: safe_remap(x, limits[col]))

        # Traitement manuel du mois (puisqu'il n'était pas dans l'encodeur LGBM)
        m = X_trans['month_mutation'].values[0]
        X_trans['month_mutation'] = safe_remap(int(m) - 1, limits['month_mutation'])

        # 3. Scaling Numérique
        X_trans[NUMERICAL_COLS] = num_scaler.transform(X_trans[NUMERICAL_COLS])

        # Création des Tenseurs
        x_num_tensor = torch.tensor(X_trans[NUMERICAL_COLS].values, dtype=torch.float32)
        x_cat_tensor = torch.tensor(X_trans[CATEGORICAL_COLS].values, dtype=torch.long)
        
        CAT_DIMS_LIST = [limits[c] for c in CATEGORICAL_COLS] 
        
        # Initialisation Modèle
        model = FTTransformer(
            n_num=len(NUMERICAL_COLS), cat_dims=CAT_DIMS_LIST, d_emb=128, 
            n_layers=3, n_heads=8, ffn_factor=4/3, attn_dropout=0.2, 
            ffn_dropout=0.1, resid_dropout=0.0
        )
        
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'ft_transformer_best_model_weights.pt'), map_location='cpu'))
        model.eval()
        
        with torch.no_grad():
            pred_log_trans = model(x_num_tensor, x_cat_tensor).item()
        
        # Dé-normalisation Target
        TARGET_MEAN, TARGET_STD = 12.4120, 0.7775 #12.0120
        pred_trans = np.exp((pred_log_trans * TARGET_STD) + TARGET_MEAN)
        
        print(f"Estimation FT-Transformer : {pred_trans:,.0f} €")
        
    except Exception as e:
        print(f"Erreur Transformer : {e}")

if __name__ == "__main__":
    main()