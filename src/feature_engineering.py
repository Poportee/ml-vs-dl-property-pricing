import pandas as pd
import numpy as np
import requests
import gzip
import io
from haversine import haversine

# 1. Données Externes
def prepare_insee_code(df):
    """Reconstitue le code INSEE à partir des codes commune et département."""
    df = df.copy()
    df['code_departement_str'] = df['code_departement'].astype(str).str.zfill(2)
    df['code_commune_str'] = df['code_commune'].astype(str).str.zfill(5)
    df['code_commune_short'] = df['code_commune_str'].str[-3:]
    df['code_departement_clean'] = df['code_departement_str'].apply(lambda x: x[1] if x.startswith('0') else x)
    df['code_insee'] = (df['code_departement_clean'] + df['code_commune_short']).astype(str).str.zfill(5)
    return df.drop(columns=['code_departement_str', 'code_commune_str', 'code_commune_short', 'code_departement_clean'])

def fetch_ban_data():
    """Télécharge et concatène les données de la Base Adresse Nationale (BAN)."""
    # URL des fichiers BAN
    ban_base_url = "https://adresse.data.gouv.fr/data/ban/adresses/latest/csv/"

    # Liste des départements (01 à 95 + DOM)
    departements = [f"{i:02d}" for i in range(1, 96) if i != 20] + ["971", "972", "973", "974", "976", "2A", "2B"]
    
    ban_frames = []
    for dep in departements:
        url = f"{ban_base_url}adresses-{dep}.csv.gz"
        try:
            r = requests.get(url)
            r.raise_for_status()
            with gzip.open(io.BytesIO(r.content), mode='rt', encoding='utf-8') as f:
                # Extraire code de voie (4 derniers caractères de 'id_fantoir')
                df_dep = pd.read_csv(f, sep=';', usecols=['id_fantoir', 'code_insee', 'lon', 'lat'])
                df_dep['code_voie'] = df_dep['id_fantoir'].astype(str).str[-4:]
                ban_frames.append(df_dep)
        except Exception:
            continue # Ignorer silencieusement les échecs pour la production
            
    ban_df = pd.concat(ban_frames, ignore_index=True)
    ban_df['code_insee'] = ban_df['code_insee'].astype(str).str.zfill(5)
    return ban_df

def fetch_communes_data():
    """Télécharge les informations des communes de France."""
    url = "https://static.data.gouv.fr/resources/communes-et-villes-de-france-en-csv-excel-json-parquet-et-feather/20250221-162232/communes-france-2025.csv"
    df = pd.read_csv(url, sep=',', engine='python', on_bad_lines='skip')
    df = df[['code_insee', 'latitude_centre', 'longitude_centre', 'densite', 'altitude_moyenne', 'population']]
    return df.rename(columns={
        'latitude_centre': 'lat_centre', 'longitude_centre': 'lon_centre',
        'densite': 'densite_commune', 'altitude_moyenne': 'altitude_moyenne_commune',
        'population': 'population_commune'
    })

def enrich_with_geodata(df, ban_df, communes_df):
    """Joint les coordonnées BAN et Communes au dataframe principal et calcule la distance."""
    # Coordonnées des voies
    voies_uniques = df[['code_insee', 'code_voie']].drop_duplicates()
    voies_coords = pd.merge(voies_uniques, ban_df[['code_insee', 'code_voie', 'lon', 'lat']], on=['code_insee', 'code_voie'], how='left')
    voies_coords = voies_coords.groupby(['code_insee', 'code_voie'], as_index=False).agg({'lat': 'mean', 'lon': 'mean'})
    
    # Infos communes
    voies_coords = pd.merge(voies_coords, communes_df, on='code_insee', how='left')
    
    # Calcul distance
    def calc_distance(row):
        if pd.notnull(row['lat']) and pd.notnull(row['lat_centre']):
            return haversine((row['lat_centre'], row['lon_centre']), (row['lat'], row['lon']))
        return np.nan
        
    voies_coords['distance_centre_ville'] = voies_coords.apply(calc_distance, axis=1)
    
    # Jointure finale
    final_df = pd.merge(
        df, 
        voies_coords[['code_insee', 'code_voie', 'lat', 'lon', 'distance_centre_ville', 'densite_commune', 'altitude_moyenne_commune', 'population_commune']],
        on=['code_insee', 'code_voie'], 
        how='left'
    )
    return final_df

# 2. Ingénierie des caractéristiques (Agrégats)
def add_postal_code_aggregates(df):
    agg_cp = df.groupby("code_postal").agg(
        prix_m2_median_cp=("prix_m2", "median"),
        prix_m2_mean_cp=("prix_m2", "mean"),
        prix_m2_std_cp=("prix_m2", "std"),
        surface_m2_median_cp=("surface_m2", "median"),
        nb_pieces_mean_cp=("nb_pieces_principales", "mean"),
        transactions_cp=("prix_m2", "count")
    ).reset_index()
    return df.merge(agg_cp, on="code_postal", how="left")

def fast_rolling_stats(group):
    dates = group["date_mutation"].astype("int64").to_numpy()
    prices = group["prix_m2"].to_numpy()
    n = len(group)
    medians = np.empty(n)
    counts = np.empty(n, dtype=np.int32)
    year_ns = np.int64(365 * 24 * 3600 * 1e9)

    for i in range(n):
        start = dates[i] - year_ns
        j = np.searchsorted(dates, start, side="left")
        window = prices[j:i+1]
        medians[i] = np.median(window)
        counts[i] = window.size

    return pd.DataFrame({
        "prix_m2_median_cp_last_12m": medians,
        "transactions_cp_last_12m": counts
    }, index=group.index)

def add_temporal_aggregates(df):
    df = df.sort_values(["code_postal", "date_mutation"]).reset_index(drop=True)
    df_stats = df[["code_postal", "date_mutation", "prix_m2"]].groupby("code_postal", group_keys=False).apply(fast_rolling_stats, include_groups=False)
    df[["prix_m2_median_cp_last_12m", "transactions_cp_last_12m"]] = df_stats
    return df

def add_combined_features(df):
    df["densite_x_population"] = df["densite_commune"] * df["population_commune"]
    df["densite_population_ratio"] = df["population_commune"] / df["densite_commune"].replace(0, 1)
    df["densite_altitude_ratio"] = df["densite_commune"] / df["altitude_moyenne_commune"].replace(0, 1)
    df["attractivite_simple"] = df["population_commune"] / df["distance_centre_ville"].replace(0, 1)
    df["distance_x_densite"] = df["distance_centre_ville"] * df["densite_commune"]
    return df

def clean_redundant_features(df):
    cols_to_drop = ['commune', 'code_commune']
    return df.drop(columns=cols_to_drop, errors='ignore')