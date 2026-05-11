# annpinguinos.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from procesador_datos import limpiar_y_preparar 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix 


# 1. Llamar al módulo (X e y YA VIENEN listos y escalados)
#X, y, le_species, df_limpio = limpiar_y_preparar("penguins.csv")
X, y, le_species, scaler, df_limpio = limpiar_y_preparar("penguins.csv")
# 2. Convertir a One-Hot (usando la 'y' que ya recibimos)
y_onehot = np.eye(3)[y]

# 3. Mezclar y dividir
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.30, random_state=None, shuffle=True)


print(f"Datos listos. Registros para entrenamiento: {len(X_train)}")

# 4. RED NEURONAL (4 neuronas, Sigmoide, Backpropagation)
def sigmoide(x): return 1 / (1 + np.exp(-x))
def sigmoide_derivada(x): return x * (1 - x)

class RedNeuronal:
    def __init__(self):
        np.random.seed(None)  # Pesos aleatorios cada vez que se ejecuta
        self.w_oculta = np.random.rand(4, 4)
        self.w_salida = np.random.rand(4, 3)
        # Tasa de aprendizaje = 0.1
        self.lr = 0.1
        self.historial_pesos = [] # <--- Para guardar el ajuste
        self.historial_error = [] # <--- Para guardar el MSE
        print("Red Neuronal inicializada con pesos aleatorios.")
        print("Pesos capa oculta:\n", self.w_oculta)
        print("Pesos capa de salida:\n", self.w_salida)
        print("¡Comenzando el entrenamiento...")
    def entrenar(self, X, y, epochs=10000):
        for i in range(epochs):
            c_oculta = sigmoide(np.dot(X, self.w_oculta))
            c_salida = sigmoide(np.dot(c_oculta, self.w_salida))

# --- GUARDAR PESOS (cada 100 épocas para no saturar memoria) ---
            if i % 100 == 0:
                self.historial_pesos.append(self.w_salida.copy().flatten())



            error_s = y - c_salida
            mse = np.mean(np.square(error_s))
            self.historial_error.append(mse)

            delta_s = error_s * sigmoide_derivada(c_salida)
            error_o = delta_s.dot(self.w_salida.T)
            delta_o = error_o * sigmoide_derivada(c_oculta)
            self.w_salida += c_oculta.T.dot(delta_s) * self.lr
            self.w_oculta += X.T.dot(delta_o) * self.lr

    def predecir(self, X):
        c_oculta = sigmoide(np.dot(X, self.w_oculta))
        return sigmoide(np.dot(c_oculta, self.w_salida))

# 5. EJECUCIÓN
nn = RedNeuronal()
nn.entrenar(X_train, y_train)

# 6. EVALUACIÓN Y MATRIZ DE CONFUSIÓN
y_pred_probs = nn.predecir(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Matriz de Confusión con Seaborn
cm = confusion_matrix(y_test_labels, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=le_species.classes_, 
            yticklabels=le_species.classes_)

plt.title('Matriz de Confusión (Datos Mezclados)')
plt.xlabel('Predicción de la Red')
plt.ylabel('Especie Real')
plt.show()

# Comparativa rápida en consola
resultados = pd.DataFrame({
    'Real': le_species.inverse_transform(y_test_labels),
    'Predicción': le_species.inverse_transform(y_pred_labels)
})
print(resultados.head(10))
# --- EVALUACIÓN FINAL ---

# 1. Obtener predicciones del set de prueba
predicciones_probs = nn.predecir(X_test)
y_pred_labels = np.argmax(predicciones_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# 2. Crear tabla comparativa con nombres reales
comparativa = pd.DataFrame({
    'Especie Real': le_species.inverse_transform(y_test_labels),
    'Predicción Red': le_species.inverse_transform(y_pred_labels)
})

# 3. Calcular si hubo éxito en cada fila
comparativa['Resultado'] = comparativa['Especie Real'] == comparativa['Predicción Red']
comparativa['Estatus'] = comparativa['Resultado'].map({True: '✅ Correcto', False: '❌ Error'})

# 4. Mostrar TODOS los datos de prueba
# Ajustamos pandas para que no recorte la salida al imprimir
pd.set_option('display.max_rows', None) 

print("\n" + "="*50)
print("   COMPARATIVA COMPLETA (SET DE PRUEBA)")
print("="*50)
print(comparativa[['Especie Real', 'Predicción Red', 'Estatus']])
print("="*50)

# 5. Calcular y mostrar % de éxito
total_pruebas = len(comparativa)
aciertos = comparativa['Resultado'].sum()
porcentaje_exito = (aciertos / total_pruebas) * 100

print(f"\nRESUMEN DE DESEMPEÑO:")
print(f"Total de pingüinos evaluados: {total_pruebas}")
print(f"Total de identificaciones correctas: {aciertos}")
print(f"Porcentaje de éxito: {porcentaje_exito:.2f}%")
print("="*50)

# Opcional: Mostrar cuáles fallaron específicamente
if aciertos < total_pruebas:
    print("\nDetalle de los errores encontrados:")
    print(comparativa[comparativa['Resultado'] == False])


import matplotlib.pyplot as plt

# Convertir el historial a un array de numpy
historial = np.array(nn.historial_pesos)

plt.figure(figsize=(10, 6))
plt.plot(historial) # Grafica todas las columnas (pesos) a la vez

plt.title('Ajuste de Pesos de la Capa de Salida (Backpropagation)')
plt.xlabel('Épocas (x100)')
plt.ylabel('Valor del Peso')
plt.grid(True, alpha=0.3)
plt.show()


import matplotlib.pyplot as plt

# Graficar la curva de pérdida (Loss Curve)
plt.figure(figsize=(10, 5))
plt.plot(nn.historial_error, color='red', linewidth=2)

plt.title('Curva de Aprendizaje: Error Cuadrático Medio (MSE)')
plt.xlabel('Épocas')
plt.ylabel('Error (MSE)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# 1. Define los datos de tu pingüino manual
# Orden: bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g
mi_pinguino = np.array([[50.0, 15.0, 220.0, 5000.0]]) # Ejemplo: Un pingüino grande

# 2. ESCALAR los datos (Muy importante: debe ser el mismo scaler de tu entrenamiento)
# Si usaste el módulo procesador_datos, asegúrate de tener acceso al objeto 'scaler'
mi_pinguino_escalado = scaler.transform(mi_pinguino)

# 3. PASAR POR LA RED
prediccion_prob = nn.predecir(mi_pinguino_escalado)
especie_index = np.argmax(prediccion_prob)

# 4. TRADUCIR EL RESULTADO
especie_final = le_species.inverse_transform([especie_index])[0]

print("\n--- PREDICCIÓN DE PINGÜINO INDIVIDUAL ---")
print(f"Medidas ingresadas: {mi_pinguino}")
print(f"Probabilidades por especie: {prediccion_prob}")
print(f"La red neuronal dice que es un: **{especie_final}**")


# --- MOSTRAR PESOS FINALES (Conocimiento de la Red) ---

# Aseguramos que los nombres de las variables coincidan
features_nombres = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
especies_nombres = le_species.classes_

print("\n" + "█"*50)
print("   ESTRUCTURA DE PESOS APRENDIDOS")
print("█"*50)

# 1. Pesos Capa Oculta
print("\n[CAPA 1] PESOS DE ENTRADA A NEURONAS OCULTAS")
print("¿Cómo afectan las medidas físicas a cada neurona interna?")
df_oculta = pd.DataFrame(nn.w_oculta, 
                         index=features_nombres, 
                         columns=['Neurona H1', 'Neurona H2', 'Neurona H3', 'Neurona H4'])
print(df_oculta)

print("\n" + "-"*50)

# 2. Pesos Capa de Salida
print("\n[CAPA 2] PESOS DE NEURONAS OCULTAS A SALIDA")
print("¿Cómo vota cada neurona interna por cada especie?")
df_salida = pd.DataFrame(nn.w_salida, 
                         index=['Neurona H1', 'Neurona H2', 'Neurona H3', 'Neurona H4'], 
                         columns=especies_nombres)
print(df_salida)
print("\n" + "█"*50)
# --- GUARDAR PESOS EN ARCHIVOS CSV ---

# 1. Guardar pesos de la Capa Oculta
df_oculta.to_csv("pesos_capa_oculta.csv")

# 2. Guardar pesos de la Capa de Salida
df_salida.to_csv("pesos_capa_salida.csv")

print("\n" + "✅ ARCHIVOS GUARDADOS:")
print("- pesos_capa_oculta.csv (Entradas -> Capa Oculta)")
print("- pesos_capa_salida.csv (Capa Oculta -> Especies)")


import pickle

# Guardar el Scaler (importante para medidas físicas)
with open('scaler_pinguinos.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Guardar el LabelEncoder (importante para los nombres de las especies)
with open('le_species.pkl', 'wb') as f:
    pickle.dump(le_species, f)

print("✅ Objetos de preprocesamiento (Scaler y LabelEncoder) guardados correctamente.")

import numpy as np

# Guardar las matrices de pesos ajustadas por el Backpropagation
np.save('pesos_ocultos.npy', nn.w_oculta)
np.save('pesos_salida.npy', nn.w_salida)

print("✅ Pesos de la red neuronal guardados en formato .npy.")




# --- GUARDAR TODO EN FORMATO CSV ---

# 1. Guardar el Scaler (Media y Desviación Estándar)
# El scaler guarda la media (mean_) y la escala (scale_) para cada característica
df_scaler = pd.DataFrame({
    'Caracteristica': features_nombres,
    'Media': scaler.mean_,
    'Desviacion_Estandar': np.sqrt(scaler.var_)
})
df_scaler.to_csv("scaler_parametros.csv", index=False)

# 2. Guardar el Encoder (Nombres de las especies y su ID)
# Guardamos la relación entre el número (0, 1, 2) y el nombre (Adelie, etc.)
df_encoder = pd.DataFrame({
    'ID': range(len(le_species.classes_)),
    'Especie': le_species.classes_
})
df_encoder.to_csv("encoder_especies.csv", index=False)

# 3. Guardar Pesos de la Capa Oculta
# (Ya lo teníamos, pero lo incluimos para completar el set)
df_oculta = pd.DataFrame(nn.w_oculta, index=features_nombres, 
                         columns=['H1', 'H2', 'H3', 'H4'])
df_oculta.to_csv("pesos_capa_oculta.csv")

# 4. Guardar Pesos de la Capa de Salida
df_salida = pd.DataFrame(nn.w_salida, index=['H1', 'H2', 'H3', 'H4'], 
                         columns=le_species.classes_)
df_salida.to_csv("pesos_capa_salida.csv")

print("\n" + "█"*50)
print("   AUDITORÍA DE ARCHIVOS GENERADOS (CSV)")
print("█"*50)
print("- scaler_parametros.csv  -> (Para normalizar datos nuevos)")
print("- encoder_especies.csv   -> (Para traducir IDs a nombres)")
print("- pesos_capa_oculta.csv  -> (Conocimiento Capa 1)")
print("- pesos_capa_salida.csv  -> (Conocimiento Capa 2)")
print("█"*50)
