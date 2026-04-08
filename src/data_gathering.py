import os
import tempfile
import zipfile
import requests
from io import BytesIO
import pandas as pd
import numpy as np
import fastparquet as fp
from fastparquet import write as fp_write

# Colonnes à conserver
KEEP = [
    "Date mutation",           # date
    "Valeur fonciere",         # prix
    "Type local",              # Appartement / Maison / Dépendance
    "Surface reelle bati",     # surface bâtie (m2)
    "Nombre pieces principales",
    "Code postal",
    "Commune",
    "Code departement",
    "Code commune",
    "Code voie",
    "Nombre de lots"
]

# Renommage en colonnes plus courtes/propres
RENAME_MAP = {
    "Date mutation":"date_mutation",
    "Valeur fonciere":"valeur_fonciere",
    "Type local":"type_local",
    "Surface reelle bati":"surface_reelle_bati",
    "Nombre pieces principales":"nb_pieces_principales",
    "Code postal":"code_postal",
    "Commune":"commune",
    "Code departement":"code_departement",
    "Code commune":"code_commune",
    "Code voie":"code_voie",
    "Nombre de lots":"nombre_de_lots"
}

# --- Fonctions Utilitaires ---
def download_to_temp(url):
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    return BytesIO(r.content)

def open_maybe_zip(fileobj):
    fileobj.seek(0)
    try:
        with zipfile.ZipFile(fileobj) as z:
            names = z.namelist()
            if not names:
                raise ValueError("ZIP vide")
            return z.open(names[0])
    except zipfile.BadZipFile:
        fileobj.seek(0)
        return fileobj

def read_csv_with_fallback(file_like, chunksize, encodings=('utf-8', 'latin-1')):
    for enc in encodings:
        try:
            file_like.seek(0)
            return pd.read_csv(
                file_like, sep='|', header=0, encoding=enc,
                dtype=str, chunksize=chunksize, low_memory=True
            )
        except Exception:
            continue
    raise RuntimeError(f"Impossible de lire le fichier avec les encodages {encodings}")

# --- Traitement des données ---
def clean_chunk(chunk):
    """Nettoie un chunk DVF et renvoie le dataframe nettoyé."""
    chunk.columns = [c.strip() if isinstance(c, str) else c for c in chunk.columns]
    
    available = [c for c in KEEP if c in chunk.columns]
    if not available:
        lower_map = {c.lower(): c for c in chunk.columns}
        available = [lower_map[d.lower()] for d in KEEP if d.lower() in lower_map]

    if not available:
        return None

    df = chunk[available].copy()
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    if 'nature_mutation' in chunk.columns:
        df['nature_mutation'] = chunk['nature_mutation'].astype(str)
        df = df[df['nature_mutation'].str.upper() == 'VENTE']

    if 'date_mutation' in df.columns:
        df['date_mutation'] = pd.to_datetime(df['date_mutation'], dayfirst=True, errors='coerce')

    if 'valeur_fonciere' in df.columns:
        df['valeur_fonciere'] = df['valeur_fonciere'].str.replace(' ', '').str.replace(',', '.')
        df['valeur_fonciere'] = pd.to_numeric(df['valeur_fonciere'], errors='coerce')

    for col in ['surface_reelle_bati', 'surface_carrez_1er_lot']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')

    if 'nb_pieces_principales' in df.columns:
        df['nb_pieces_principales'] = pd.to_numeric(df['nb_pieces_principales'], errors='coerce')

    if 'code_postal' in df.columns:
        df['code_postal'] = df['code_postal'].astype(str).str.zfill(5).str.slice(0, 5)

    if 'code_commune' in df.columns:
        df['code_commune'] = df['code_commune'].astype(str).str.zfill(5)

    if 'code_voie' in df.columns:
        df['code_voie'] = df['code_voie'].astype(str).str.zfill(4)

    df['surface_m2'] = df['surface_reelle_bati'] if 'surface_reelle_bati' in df.columns else np.nan

    df['prix_m2'] = np.nan
    mask = (df.get('valeur_fonciere', pd.Series()).notna() & df['surface_m2'].notna() & (df['surface_m2'] > 0))
    df.loc[mask, 'prix_m2'] = df.loc[mask, 'valeur_fonciere'] / df.loc[mask, 'surface_m2']

    valid = pd.Series(True, index=df.index)
    if 'valeur_fonciere' in df.columns:
        valid &= df['valeur_fonciere'] > 0
    valid &= df['surface_m2'].fillna(0) >= 4
    if 'prix_m2' in df.columns:
        valid &= ~((df['prix_m2'] > 20000) | (df['prix_m2'] < 50))

    return df[valid]

def sanitize_for_parquet(df):
    df2 = pd.DataFrame()  
    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.PeriodDtype):
            df2[col] = s.astype("datetime64[ns]")  
        elif pd.api.types.is_object_dtype(s):
            df2[col] = s.astype("string")
        elif pd.api.types.is_datetime64_any_dtype(s):
            df2[col] = s.dt.floor("ms")
        else:
            df2[col] = s
    return df2

def process_zip_url(url, year, outdir, chunksize=200_000):
    print(f"[{year}] Téléchargement et traitement : {url}")
    fileobj = download_to_temp(url)
    member_file = open_maybe_zip(fileobj)
    reader = read_csv_with_fallback(member_file, chunksize)

    tmp_files = []
    for i, chunk in enumerate(reader):
        cleaned = clean_chunk(chunk)
        if cleaned is None or cleaned.empty:
            continue

        cleaned["source_year"] = int(year)
        df2 = sanitize_for_parquet(cleaned)
        path = os.path.join(tempfile.gettempdir(), f"dvf_chunk_{year}_{i}.parquet")
        fp_write(path, df2, write_index=False)
        tmp_files.append(path)

    if not tmp_files:
        return None

    df_year = pd.concat([fp.ParquetFile(p).to_pandas() for p in tmp_files], ignore_index=True)
    out_path = os.path.join(outdir, "per_year", f"dvf_clean_{year}.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fp_write(out_path, df_year, write_index=False)

    for p in tmp_files:
        try: os.remove(p)
        except: pass

    return out_path

def merge_years(parquet_paths, outpath):
    dfs = [fp.ParquetFile(p).to_pandas() for p in parquet_paths]
    df = pd.concat(dfs, ignore_index=True)
    fp_write(outpath, df, write_index=False)
    return outpath