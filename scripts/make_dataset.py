import os
import sys
import pandas as pd

# Ajout du dossier racine au path pour importer src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_gathering import process_zip_url, merge_years
from src.feature_engineering import (
    prepare_insee_code, fetch_ban_data, fetch_communes_data,
    enrich_with_geodata, add_postal_code_aggregates,
    add_temporal_aggregates, add_combined_features,
    clean_redundant_features
)

def main():
    # Configuration 
    DATA_DIR = "./data/processed"
    URLS_DVF = {
        2025 : "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20251018-234857/valeursfoncieres-2024.txt.zip",
        2024 : "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20251018-234851/valeursfoncieres-2023.txt.zip",
        2023 : "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20251018-234844/valeursfoncieres-2022.txt.zip",
        2022 : "https://static.data.gouv.fr/resources/demandes-de-valeurs-foncieres/20251018-234836/valeursfoncieres-2021.txt.zip",
        2021 : "http://data.cquest.org/dgfip_dvf/202104/valeursfoncieres-2020.txt",
        2020 : "http://data.cquest.org/dgfip_dvf/202104/valeursfoncieres-2019.txt",
        2019 : "http://data.cquest.org/dgfip_dvf/202104/valeursfoncieres-2018.txt",
        2018 : "http://data.cquest.org/dgfip_dvf/202104/valeursfoncieres-2017.txt",
        2017 : "http://data.cquest.org/dgfip_dvf/202104/valeursfoncieres-2016.txt",
        2016 : "http://data.cquest.org/dgfip_dvf/201904/valeursfoncieres-2015.txt",
        2015 : "http://data.cquest.org/dgfip_dvf/201904/valeursfoncieres-2014.txt"
    }
    FINAL_OUTPUT = os.path.join(DATA_DIR, "final_dataset.parquet")

    print("Démarrage du pipeline de préparation des données...")

    # 1. Collecte et Nettoyage initial
    path_list = []
    for year, url in URLS_DVF.items():
        p = process_zip_url(url, year, DATA_DIR)
        if p:
            path_list.append(p)

    print("Fusion des fichiers annuels...")
    merged_path = os.path.join(DATA_DIR, "dvf_merged_raw.parquet")
    df = pd.read_parquet(merge_years(path_list, merged_path))

    # 2. Préparation des codes géographiques
    print("Préparation des codes INSEE...")
    df = prepare_insee_code(df)

    # 3. Récupération des données externes
    print("Téléchargement des données BAN (Base Adresse Nationale)...")
    ban_df = fetch_ban_data()
    
    print("Téléchargement des données Communes...")
    communes_df = fetch_communes_data()

    # 4. Enrichissement Géographique
    print("Calcul des distances et densités...")
    df = enrich_with_geodata(df, ban_df, communes_df)

    # 5. Feature Engineering (Agrégats)
    print("Calcul des agrégats par code postal...")
    df = add_postal_code_aggregates(df)
    
    print("Calcul des statistiques temporelles (Rolling 12m)...")
    df = add_temporal_aggregates(df)
    
    print("Création des features combinées...")
    df = add_combined_features(df)

    # 6. Nettoyage final
    print("Nettoyage final des colonnes redondantes...")
    df = clean_redundant_features(df)

    # 7. Sauvegarde
    print(f"auvegarde du dataset final ({len(df)} lignes) -> {FINAL_OUTPUT}")
    df.to_parquet(FINAL_OUTPUT, index=False)
    
    print("Pipeline terminé avec succès !")

if __name__ == "__main__":
    main()