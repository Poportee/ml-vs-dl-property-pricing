import os
import pandas as pd
import fastparquet as fp

def main():
    print("--- Génération des statistiques par Code Postal ---")
    
    # 1. Charger les données enrichies (qui contiennent déjà les stats)
    data_path = "./data/processed/final_dataset.parquet"
    if not os.path.exists(data_path):
        print(f"Erreur : Le fichier {data_path} est introuvable.")
        return

    df = fp.ParquetFile(data_path).to_pandas()
    
    # S'assurer que les données sont triées par date pour garder la stat la plus récente
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    df = df.sort_values("date_mutation")

    # 2. Sélectionner les colonnes de statistiques
    stats_cols = [
        'code_postal', 'prix_m2_median_cp', 'prix_m2_mean_cp', 'prix_m2_std_cp', 
        'surface_m2_median_cp', 'nb_pieces_mean_cp', 'transactions_cp', 
        'prix_m2_median_cp_last_12m', 'transactions_cp_last_12m'
    ]
    
    # 3. Ne garder que la dernière ligne (la plus récente) pour chaque code postal
    cp_stats = df[stats_cols].drop_duplicates(subset=['code_postal'], keep='last')
    
    # 4. Sauvegarder
    out_dir = "./data/processed"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "cp_stats.parquet")
    
    cp_stats.to_parquet(out_path, index=False)
    print(f"Succès ! Statistiques sauvegardées pour {len(cp_stats)} codes postaux dans {out_path}.")

if __name__ == "__main__":
    main()