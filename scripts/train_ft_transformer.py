import os
import sys
import copy
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import fastparquet as fp
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Import des modules locaux
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.evaluation import evaluate_by_price_range
from src.ft_transformer_net import FTTransformer

D_EMB = 128  
N_HEADS = 8
N_LAYERS = 3 
FFN_FACTOR = 4/3
ATTENTION_DROPOUT = 0.2
FFN_DROPOUT = 0.1
RESIDUAL_DROPOUT = 0.0
LR = 1e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 256
N_EPOCHS = 50
PATIENCE = 10


def remap_categorical_indices(X_cat):
    """Remappe les catégories pour réserver l'index 0 aux valeurs inconnues (-1)."""
    X_cat = X_cat.astype(int)
    unknown_mask = (X_cat == -1)
    X_cat[unknown_mask] = 0        
    X_cat[~unknown_mask] += 1      
    return X_cat

def main():
    print("--- 1. Chargement des données enrichies ---")
    df = fp.ParquetFile("./data/processed/final_dataset.parquet").to_pandas()
    
    # Gestion des dates et filtrage des valeurs aberrantes
    df["date_mutation"] = pd.to_datetime(df["date_mutation"], errors="coerce")
    df = df[df["valeur_fonciere"] < 1_000_000].copy()
    df = df.sort_values("date_mutation")
    df["target"] = np.log(df["valeur_fonciere"])
    df['month_mutation'] = df['date_mutation'].dt.month

    # Traitement des NaNs/Cohérence des colonnes numériques (avant le split)
    df['nombre_de_lots'] = pd.to_numeric(df['nombre_de_lots'], errors='coerce').fillna(1)
    df['code_insee'] = pd.to_numeric(df['code_insee'], errors='coerce').fillna(0)
    df['nb_pieces_principales'] = pd.to_numeric(df['nb_pieces_principales'], errors='coerce').fillna(1)

    CATEGORICAL_COLS = ['type_local', 'code_postal', 'code_departement', 'code_voie', 'month_mutation']
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
    NUMERICAL_COLS = [c for c in features if c not in CATEGORICAL_COLS]

    print("--- 2. Split Temporel Train / Test ---")
    train_mask = df["date_mutation"] < "2023-11-01"
    X_train, y_train = df.loc[train_mask, features].copy(), df.loc[train_mask, "target"].copy()
    X_test, y_test = df.loc[~train_mask, features].copy(), df.loc[~train_mask, "target"].copy()

    print("--- 3. Preprocessing (Encodage, Imputation, Scaling) ---")
    # Encodage catégoriel
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train.loc[:, CATEGORICAL_COLS] = encoder.fit_transform(X_train[CATEGORICAL_COLS])
    X_test.loc[:, CATEGORICAL_COLS] = encoder.transform(X_test[CATEGORICAL_COLS])

    X_train.loc[:, CATEGORICAL_COLS] = remap_categorical_indices(X_train[CATEGORICAL_COLS].values)
    X_test.loc[:, CATEGORICAL_COLS] = remap_categorical_indices(X_test[CATEGORICAL_COLS].values)
    CAT_DIMS = {col: X_train[col].max() + 1 for col in CATEGORICAL_COLS}

    # Imputation des NaNs restants (ex: communes sans géoloc)
    COLUMNS_TO_IMPUTE = [
        'lat', 'lon', 'distance_centre_ville', 'densite_commune', 'altitude_moyenne_commune', 
        'population_commune', 'densite_x_population', 'densite_population_ratio', 
        'densite_altitude_ratio', 'attractivite_simple', 'distance_x_densite'
    ]
    
    imp_map_train = X_train.groupby('code_insee')[COLUMNS_TO_IMPUTE].transform('median')
    imp_map_test = X_test.groupby('code_insee')[COLUMNS_TO_IMPUTE].transform('median')

    X_train.loc[:, COLUMNS_TO_IMPUTE] = X_train[COLUMNS_TO_IMPUTE].fillna(imp_map_train)
    X_test.loc[:, COLUMNS_TO_IMPUTE] = X_test[COLUMNS_TO_IMPUTE].fillna(imp_map_test)
    
    global_median = X_train[COLUMNS_TO_IMPUTE].median()
    X_train.loc[:, COLUMNS_TO_IMPUTE] = X_train[COLUMNS_TO_IMPUTE].fillna(global_median)
    X_test.loc[:, COLUMNS_TO_IMPUTE] = X_test[COLUMNS_TO_IMPUTE].fillna(global_median)

    # Scaling
    num_scaler = StandardScaler()
    X_train.loc[:, NUMERICAL_COLS] = num_scaler.fit_transform(X_train[NUMERICAL_COLS])
    X_test.loc[:, NUMERICAL_COLS] = num_scaler.transform(X_test[NUMERICAL_COLS])

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

    os.makedirs("./models", exist_ok=True)
    joblib.dump(num_scaler, './models/num_scaler_ft.pkl')
    joblib.dump(encoder, './models/encoder_categorielles_ft.pkl')

    print("--- 4. Préparation des Tensors PyTorch ---")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_num_t = torch.tensor(X_train[NUMERICAL_COLS].values, dtype=torch.float32)
    X_train_cat_t = torch.tensor(X_train[CATEGORICAL_COLS].values, dtype=torch.long)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)

    X_test_num_t = torch.tensor(X_test[NUMERICAL_COLS].values, dtype=torch.float32)
    X_test_cat_t = torch.tensor(X_test[CATEGORICAL_COLS].values, dtype=torch.long)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

    train_data = TensorDataset(X_train_num_t, X_train_cat_t, y_train_t)
    test_data = TensorDataset(X_test_num_t, X_test_cat_t, y_test_t)

    train_len = int(0.8 * len(train_data))
    val_len = len(train_data) - train_len
    train_subset, val_subset = random_split(train_data, [train_len, val_len])

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    print("--- 5. Initialisation du FT-Transformer ---")
    model = FTTransformer(
        n_num=len(NUMERICAL_COLS), 
        cat_dims=list(CAT_DIMS.values()), 
        d_emb=128, 
        n_layers=N_LAYERS, 
        n_heads=N_HEADS, 
        ffn_factor=FFN_FACTOR, 
        attn_dropout=ATTENTION_DROPOUT, 
        ffn_dropout=FFN_DROPOUT, 
        resid_dropout=RESIDUAL_DROPOUT
    ).to(DEVICE)

    no_decay = ['bias', 'LayerNorm.weight', 'feature_tokenizer.num_biases', 'feature_tokenizer.cat_biases', 'cls_token']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-5},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_weights = None

    print("--- 6. Entraînement ---")
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0
        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x_num, x_cat)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
                output = model(x_num, x_cat)
                val_loss += criterion(output, y).item() * len(y)
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_subset)
        print(f"Epoch {epoch+1}/{N_EPOCHS} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:
                print("Early stopping déclenché.")
                break

    if best_model_weights:
        model.load_state_dict(best_model_weights)
        torch.save(model.state_dict(), './models/ft_transformer_best.pt')

    print("--- 7. Prédictions & Évaluation ---")
    model.eval()
    test_predictions, test_targets = [], []
    with torch.no_grad():
        for x_num, x_cat, y in test_loader:
            output = model(x_num.to(DEVICE), x_cat.to(DEVICE))
            test_predictions.append(output.cpu().numpy())
            test_targets.append(y.cpu().numpy())

    # Dé-normalisation pour obtenir le log(prix)
    y_pred_log_raw = (np.concatenate(test_predictions).flatten() * target_scaler.scale_[0]) + target_scaler.mean_[0]
    y_true_log_raw = (np.concatenate(test_targets).flatten() * target_scaler.scale_[0]) + target_scaler.mean_[0]

    # Exponentielle pour revenir aux prix en Euros
    evaluate_by_price_range(np.exp(y_true_log_raw), np.exp(y_pred_log_raw), "FT-Transformer")

if __name__ == "__main__":
    main()