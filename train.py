import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import xgboost as xgb

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calcule la distance en km entre deux points géographiques"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

print("=== ÉTAPE 1: CHARGEMENT DES DONNÉES ===")
file_path = 'Earthquake_data_cleaned.csv'
df = pd.read_csv(file_path)

print("\n=== ÉTAPE 2: NETTOYAGE ===")
def nettoyer_donnees(df):
    df_clean = df.drop_duplicates().copy()
    colonnes_critiques = ['event_id', 'eq_magnitude', 'eq_latitude', 'eq_longitude']
    df_clean = df_clean.dropna(subset=colonnes_critiques)
    df_clean = df_clean[
        (df_clean['eq_latitude'] >= -90) & (df_clean['eq_latitude'] <= 90) &
        (df_clean['eq_longitude'] >= -180) & (df_clean['eq_longitude'] <= 180)
    ]
    return df_clean

df_clean = nettoyer_donnees(df)

print("\n=== ÉTAPE 3: FORMULES PHYSIQUES ===")
def estimer_epicentre(gdf):
    gdf = gdf.copy().dropna(subset=["cdi"])
    if gdf.empty: return Point(0, 0)
    weights = gdf["cdi"]
    x = np.average(gdf.geometry.x, weights=weights)
    y = np.average(gdf.geometry.y, weights=weights)
    return Point(x, y)

def magnitude_epicentre(gdf, epicentre):
    distances = gdf.geometry.distance(epicentre)
    if (distances == 0).all(): return np.average(gdf["cdi"])
    weights = 1 / (distances + 1e-6)
    cdi_mean = np.average(gdf["cdi"], weights=weights)
    dist_mean = np.average(distances * 111, weights=weights)
    return (cdi_mean + 3.29 + 0.0206 * dist_mean) / 1.68

print("\n=== ÉTAPE 4: PRÉPROCESSING ML ===")
df_points = gpd.GeoDataFrame(df_clean, geometry=gpd.points_from_xy(df_clean.dyfi_longitude, df_clean.dyfi_latitude))
df_points.crs = "EPSG:4326"

print("Calcul des agrégations par événement...")
# Correction du warning Pandas : include_groups=False pour éviter d'inclure la colonne de groupement 'event_id'
df_agg = df_points.groupby('event_id').apply(lambda group: {
    'responses': len(group),
    'cdi_max': group['dyfi_cdi'].max(),
    'cdi_avg': group['dyfi_cdi'].mean(),
    'lat_median': group['dyfi_latitude'].median(),
    'lon_median': group['dyfi_longitude'].median(),
    'lat_max_cdi': group.loc[group['dyfi_cdi'].idxmax(), 'dyfi_latitude'],
    'lon_max_cdi': group.loc[group['dyfi_cdi'].idxmax(), 'dyfi_longitude'],
    'lat_range': group['dyfi_latitude'].max() - group['dyfi_latitude'].min(),
    'lon_range': group['dyfi_longitude'].max() - group['dyfi_longitude'].min(),
    'eq_latitude': group['eq_latitude'].iloc[0],
    'eq_longitude': group['eq_longitude'].iloc[0],
    'eq_magnitude': group['eq_magnitude'].iloc[0],
    'geometry_list': group['geometry'].tolist(),
    'cdi_list': group['dyfi_cdi'].tolist()
}, include_groups=False).to_dict()

print(f"succès : {len(df_agg)} séismes agrégés et prêts pour l'analyse.")

estimated_data = []
for eid, row in df_agg.items():
    tmp_gdf = gpd.GeoDataFrame({'cdi': row['cdi_list']}, geometry=row['geometry_list'])
    epi = estimer_epicentre(tmp_gdf)
    mag_ph = magnitude_epicentre(tmp_gdf, epi)
    
    weights = tmp_gdf['cdi']
    lat_w_std = np.sqrt(np.average((tmp_gdf.geometry.y - epi.y)**2, weights=weights)) if len(tmp_gdf) > 1 else 0
    lon_w_std = np.sqrt(np.average((tmp_gdf.geometry.x - epi.x)**2, weights=weights)) if len(tmp_gdf) > 1 else 0

    estimated_data.append({
        'event_id': eid,
        'mag_phys': mag_ph, 'lat_phys': epi.y, 'lon_phys': epi.x,
        'true_mag': row['eq_latitude'], # placeholder, will use dict below
        'data': row,
        'lat_w_std': lat_w_std, 'lon_w_std': lon_w_std
    })

df_ml = pd.DataFrame(estimated_data).set_index('event_id')
# Re-extract true values properly
df_ml['true_mag'] = df_ml['data'].apply(lambda x: x['eq_magnitude'])
df_ml['true_lat'] = df_ml['data'].apply(lambda x: x['eq_latitude'])
df_ml['true_lon'] = df_ml['data'].apply(lambda x: x['eq_longitude'])

# Engineering de "Contraste" (Deltas explicites)
df_ml['lat_diff_max'] = df_ml['lat_phys'] - df_ml['data'].apply(lambda x: x['lat_max_cdi'])
df_ml['lon_diff_max'] = df_ml['lon_phys'] - df_ml['data'].apply(lambda x: x['lon_max_cdi'])
df_ml['lat_diff_med'] = df_ml['lat_phys'] - df_ml['data'].apply(lambda x: x['lat_median'])
df_ml['lon_diff_med'] = df_ml['lon_phys'] - df_ml['data'].apply(lambda x: x['lon_median'])

df_ml['log_resp'] = np.log1p(df_ml['data'].apply(lambda x: x['responses']))
df_ml['dispersion'] = np.sqrt(df_ml['lat_w_std']**2 + df_ml['lon_w_std']**2)
df_ml['cos_lat'] = np.cos(np.radians(df_ml['lat_phys']))
df_ml['cdi_max'] = df_ml['data'].apply(lambda x: x['cdi_max'])
df_ml['cdi_avg'] = df_ml['data'].apply(lambda x: x['cdi_avg'])

# Cibles KM
df_ml['target_n_km'] = (df_ml['true_lat'] - df_ml['lat_phys']) * 111.32
df_ml['target_e_km'] = (df_ml['true_lon'] - df_ml['lon_phys']) * 111.32 * df_ml['cos_lat']
df_ml['target_mag_err'] = df_ml['true_mag'] - df_ml['mag_phys']

# Filtrage
df_train = df_ml[df_ml['data'].apply(lambda x: x['responses']) >= 3]

print("\n=== ÉTAPE 5: ENTRAÎNEMENT FINAL (DOUBLE OPTIMISATION) ===")

features_mag = ['mag_phys', 'lat_phys', 'lon_phys', 'log_resp', 'cdi_max', 'cdi_avg', 'dispersion']
features_loc = ['mag_phys', 'lat_phys', 'lon_phys', 'lat_diff_max', 'lon_diff_max', 'lat_diff_med', 'lon_diff_med', 'dispersion', 'log_resp', 'cos_lat']

X_tr, X_te = train_test_split(df_train, test_size=0.2, random_state=42)

# --- CONFIGURATION OPTIMALE MAGNITUDE (~50% d'amélioration) ---
xgb_config_mag = {
    'n_estimators': 500,
    'learning_rate': 0.03,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror', # Meilleur pour la magnitude
    'random_state': 42
}

# --- CONFIGURATION OPTIMALE LOCALISATION (88 km d'erreur) ---
xgb_config_loc = {
    'n_estimators': 600,
    'learning_rate': 0.02,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:absoluteerror', # Meilleur pour la localisation robuste
    'random_state': 42
}

print("Entraînement des modèles optimisés par tâche...")
model_m = xgb.XGBRegressor(**xgb_config_mag).fit(X_tr[features_mag], X_tr['target_mag_err'])
model_n = xgb.XGBRegressor(**xgb_config_loc).fit(X_tr[features_loc], X_tr['target_n_km'])
model_e = xgb.XGBRegressor(**xgb_config_loc).fit(X_tr[features_loc], X_tr['target_e_km'])

# Évaluation
pred_m_tr = model_m.predict(X_tr[features_mag])
pred_n_tr = model_n.predict(X_tr[features_loc])
pred_e_tr = model_e.predict(X_tr[features_loc])



pred_m_te = model_m.predict(X_te[features_mag])
pred_n_te = model_n.predict(X_te[features_loc])
pred_e_te = model_e.predict(X_te[features_loc])



# Reconstruction
final_m_te = X_te['mag_phys'] + pred_m_te
final_la_te = X_te['lat_phys'] + (pred_n_te / 111.32)
final_lo_te = X_te['lon_phys'] + (pred_e_te / (111.32 * X_te['cos_lat']))

final_m_tr = X_tr['mag_phys'] + pred_m_tr
final_la_tr = X_tr['lat_phys'] + (pred_n_tr / 111.32)
final_lo_tr = X_tr['lon_phys'] + (pred_e_tr / (111.32 * X_tr['cos_lat']))



# Métriques Magnitude
mae_mag_ia_te = mean_absolute_error(X_te['true_mag'], final_m_te)
mae_mag_ia_tr = mean_absolute_error(X_tr['true_mag'], final_m_tr)



# Métriques Localisation (Haversine)
err_ia_te = [haversine_distance(X_te['true_lat'].iloc[i], X_te['true_lon'].iloc[i], final_la_te.iloc[i], final_lo_te.iloc[i]) for i in range(len(X_te))]
err_ia_tr = [haversine_distance(X_tr['true_lat'].iloc[i], X_tr['true_lon'].iloc[i], final_la_tr.iloc[i], final_lo_tr.iloc[i]) for i in range(len(X_tr))]



# Physique seule (pour rappel)
err_ph_te = [haversine_distance(X_te['true_lat'].iloc[i], X_te['true_lon'].iloc[i], X_te['lat_phys'].iloc[i], X_te['lon_phys'].iloc[i]) for i in range(len(X_te))]



print(f"\n --- DIAGNOSTIC DU SURAPPRENTISSAGE ---")
print(f"MAGNITUDE :")
print(f" MAE Train : {mae_mag_ia_tr:.4f}")
print(f" MAE Test : {mae_mag_ia_te:.4f}")
print(f" Ecart : {abs(mae_mag_ia_tr - mae_mag_ia_te):.4f}")



print(f"\nLOCALISATION :")
print(f" Erreur Moyenne Train : {np.mean(err_ia_tr):.2f} km")
print(f" Erreur Moyenne Test : {np.mean(err_ia_te):.2f} km")
print(f" Ecart : {abs(np.mean(err_ia_tr) - np.mean(err_ia_te)):.2f} km")



print(f"\n --- PERFORMANCE vs PHYSIQUE (SUR TEST) ---")
print(f"Amélioration Magnitude : {((mean_absolute_error(X_te['true_mag'], X_te['mag_phys']) - mae_mag_ia_te)/mean_absolute_error(X_te['true_mag'], X_te['mag_phys']))*100:+.1f}%")
print(f"Amélioration Localisation : {((np.mean(err_ph_te) - np.mean(err_ia_te))/np.mean(err_ph_te))*100:+.1f}%")

print("\n=== ÉTAPE 6: SAUVEGARDE DES MODÈLES ===")
import os
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model_m, 'models/model_magnitude.pkl')
joblib.dump(model_n, 'models/model_latitude_km.pkl')
joblib.dump(model_e, 'models/model_longitude_km.pkl')
print("Modèles sauvegardés avec succès dans le dossier 'models/'.")
