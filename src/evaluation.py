import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_by_price_range(y_true_eur, y_pred_eur, model_name="Modèle"):
    """Calcule et affiche les métriques MAE, RMSE et R² globales et par tranche de prix."""
    mae = mean_absolute_error(y_true_eur, y_pred_eur)
    rmse = np.sqrt(mean_squared_error(y_true_eur, y_pred_eur))
    r2 = r2_score(y_true_eur, y_pred_eur)

    print("\n" + "="*50)
    print(f"--- Évaluation Finale : {model_name} ---")
    print("="*50)
    print("--- Performance Globale ---")
    print(f"    MAE  : {mae:,.0f} €")
    print(f"    RMSE : {rmse:,.0f} €")
    print(f"    R²   : {r2:.4f}")

    ranges = [
        (0, 200000, "0-200k€"),
        (200000, 500000, "200k-500k€"),
        (500000, 1000000, "500k-1M€"),
    ]

    n_total = len(y_true_eur)
    print("\n--- Évaluation par Tranche de Prix ---")
    
    for min_p, max_p, label in ranges:
        mask = (y_true_eur >= min_p) & (y_true_eur < max_p)
        y_true_range = y_true_eur[mask]
        y_pred_range = y_pred_eur[mask]

        if len(y_true_range) > 0:
            mae_bin = mean_absolute_error(y_true_range, y_pred_range)
            rmse_bin = np.sqrt(mean_squared_error(y_true_range, y_pred_range))
            proportion = len(y_true_range) / n_total
            
            print(f"--- {label} (proportion : {proportion:.2%}) ---")
            print(f"    MAE  : {mae_bin:,.0f} €")
            print(f"    RMSE : {rmse_bin:,.0f} €")
        else:
            print(f"--- {label} --- (AUCUNE DONNÉE)")