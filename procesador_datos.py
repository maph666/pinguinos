# procesador_datos.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def limpiar_y_preparar(ruta_csv):
    df = pd.read_csv(ruta_csv).dropna().reset_index(drop=True)
    
    # Limpieza de outliers (Z-score)
    features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    for col in features:
        z = (df[col] - df[col].mean()) / df[col].std()
        df = df[abs(z) < 3]
    df = df.reset_index(drop=True)

    # Escalado (Guardamos el objeto para usarlo después)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Codificación
    le_species = LabelEncoder()
    y_encoded = le_species.fit_transform(df['species'])
    
    # RETORNAMOS TAMBIÉN EL SCALER
    return X_scaled, y_encoded, le_species, scaler, df
