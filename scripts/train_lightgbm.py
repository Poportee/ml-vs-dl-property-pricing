import os
import sys
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
import fastparquet as fp
from sklearn.preprocessing import OrdinalEncoder

# Import des modules d'évaluation locaux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation import evaluate_by_price_range

def main():
    print("--- 1. Chargement des données enrichies ---")
    # Fichier issu de feature_engineering.py
    df = fp.ParquetFile("./data/processed/final_dataset.parquet").to_pandas()
    
    # Filtrage et création de la target
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    df = df[df["valeur_fonciere"] < 1_000_000].copy()
    df = df.sort_values("date_mutation")
    df["target"] = np.log(df["valeur_fonciere"])
    
    # Feature supplémentaire
    df['month_mutation'] = df['date_mutation'].dt.month

    # Nettoyage final des types pour LightGBM
    df['nombre_de_lots'] = pd.to_numeric(df['nombre_de_lots'], errors='coerce').fillna(1)
    df['code_insee'] = pd.to_numeric(df['code_insee'], errors='coerce').fillna(0)
    df['nb_pieces_principales'] = pd.to_numeric(df['nb_pieces_principales'], errors='coerce').fillna(1)

    features = [
        'type_local', 'surface_reelle_bati', 'nb_pieces_principales', 'code_postal', 
        'code_departement', 'code_voie', 'nombre_de_lots', 'surface_m2', 'source_year', 
        'code_insee', 'lat', 'lon', 'distance_centre_ville', 'densite_commune', 
        'altitude_moyenne_commune', 'population_commune','prix_m2_median_cp','prix_m2_mean_cp',
        'prix_m2_std_cp', 'surface_m2_median_cp','nb_pieces_mean_cp','transactions_cp',
        'prix_m2_median_cp_last_12m', 'transactions_cp_last_12m','densite_x_population',
        'densite_population_ratio', 'densite_altitude_ratio','attractivite_simple',
        'distance_x_densite', 'month_mutation'
    ]

    print("--- 2. Split Temporel Train / Test ---")
    train_mask = df["date_mutation"] < "2023-11-01"
    X_train, y_train = df.loc[train_mask, features].copy(), df.loc[train_mask, "target"].copy()
    X_test, y_test = df.loc[~train_mask, features].copy(), df.loc[~train_mask, "target"].copy()

    print(f"Taille train : {len(X_train)}")
    print(f"Taille test  : {len(X_test)}")

    print("--- 3. Encodage des variables catégorielles ---")
    categorical_cols = ['type_local', 'code_postal', 'code_departement', 'code_voie']
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    
    X_train.loc[:, categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    X_test.loc[:, categorical_cols] = encoder.transform(X_test[categorical_cols])

    print("--- 4. Entraînement du modèle LightGBM ---")
    params = {
        "objective": "regression", 
        "metric": "mae", 
        "boosting_type": "gbdt",
        "num_leaves": 64, 
        "learning_rate": 0.05, 
        "n_estimators": 4000,
        "max_depth": -1, 
        "subsample": 0.8, 
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1, 
        "reg_lambda": 0.1, 
        "verbose": -1
    }

    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_valid = lgb.Dataset(X_test, y_test)

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(stopping_rounds=200)]
    )

    print("--- 5. Prédictions et Évaluation ---")
    y_pred = np.exp(model.predict(X_test))
    y_true = np.exp(y_test)
    evaluate_by_price_range(y_true, y_pred, "LightGBM")

    print("--- 6. Sauvegarde des artefacts ---")
    os.makedirs("./models", exist_ok=True)
    joblib.dump(model, "./models/modele_lightgbm_final.pkl")
    joblib.dump(encoder, "./models/encoder_categorielles_lgbm.pkl")
    print("Sauvegarde terminée avec succès.")

if __name__ == "__main__":
    main()