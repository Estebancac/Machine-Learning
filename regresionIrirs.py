import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Configurar matplotlib para español
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

print("=== ANÁLISIS DE REGRESIÓN LINEAL - DATASET IRIS ===")
print("Creado por: Análisis de Datos en Español\n")

# 1. Cargar el dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=['Longitud Sépalo', 'Ancho Sépalo', 
                                      'Longitud Pétalo', 'Ancho Pétalo'])

print("📊 Información del Dataset:")
print(f"Tamaño: {df.shape[0]} muestras, {df.shape[1]} variables")
print("\nPrimeras 5 filas:")
print(df.head())

# 2. MODELO SIMPLE: Longitud Sépalo → Longitud Pétalo
print("\n" + "="*50)
print("🔵 MODELO DE REGRESIÓN LINEAL SIMPLE")
print("="*50)

X_simple = df[['Longitud Sépalo']].values
y_simple = df['Longitud Pétalo'].values

correlacion = np.corrcoef(X_simple.flatten(), y_simple)[0,1]
print(f"Variable independiente: Longitud del Sépalo")
print(f"Variable dependiente: Longitud del Pétalo")
print(f"Correlación: {correlacion:.3f} (relación muy fuerte)")

# Dividir datos
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y_simple, test_size=0.3, random_state=42)

# Entrenar modelo simple
modelo_simple = LinearRegression()
modelo_simple.fit(X_train_simple, y_train_simple)
y_pred_simple = modelo_simple.predict(X_test_simple)

# Métricas modelo simple
r2_simple = r2_score(y_test_simple, y_pred_simple)
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)

print(f"\n📈 Resultados:")
print(f"Ecuación: Longitud Pétalo = {modelo_simple.coef_[0]:.3f} × Longitud Sépalo + {modelo_simple.intercept_:.3f}")
print(f"R² = {r2_simple:.3f} (explica {r2_simple*100:.1f}% de la variabilidad)")
print(f"Error cuadrático medio = {mse_simple:.3f}")

# 3. MODELO MÚLTIPLE: Todas las variables → Longitud Pétalo
print("\n" + "="*50)
print("🟠 MODELO DE REGRESIÓN LINEAL MÚLTIPLE") 
print("="*50)

X_multiple = df[['Longitud Sépalo', 'Ancho Sépalo', 'Ancho Pétalo']].values
y_multiple = df['Longitud Pétalo'].values

print("Variables independientes: Longitud Sépalo + Ancho Sépalo + Ancho Pétalo")
print("Variable dependiente: Longitud Pétalo")

# Estandarizar variables (importante para regresión múltiple)
scaler = StandardScaler()
X_multiple_scaled = scaler.fit_transform(X_multiple)

# Dividir datos
X_train_mult, X_test_mult, y_train_mult, y_test_mult = train_test_split(
    X_multiple_scaled, y_multiple, test_size=0.3, random_state=42)

# Entrenar modelo múltiple
modelo_multiple = LinearRegression()
modelo_multiple.fit(X_train_mult, y_train_mult)
y_pred_mult = modelo_multiple.predict(X_test_mult)

# Métricas modelo múltiple
r2_multiple = r2_score(y_test_mult, y_pred_mult)
mse_multiple = mean_squared_error(y_test_mult, y_pred_mult)

print(f"\n📈 Resultados:")
print(f"R² = {r2_multiple:.3f} (explica {r2_multiple*100:.1f}% de la variabilidad)")
print(f"Error cuadrático medio = {mse_multiple:.3f}")

# Importancia de variables
variables = ['Longitud Sépalo', 'Ancho Sépalo', 'Ancho Pétalo']
coeficientes = modelo_multiple.coef_
importancia = list(zip(variables, abs(coeficientes)))
importancia.sort(key=lambda x: x[1], reverse=True)

print(f"\n🎯 Importancia de las variables:")
for i, (var, imp) in enumerate(importancia, 1):
    print(f"{i}. {var}: {imp:.3f}")

# ===============================================
# VISUALIZACIONES EN ESPAÑOL
# ===============================================

# GRÁFICA 1: Regresión Lineal Simple
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Datos de entrenamiento y prueba
plt.scatter(X_train_simple, y_train_simple, alpha=0.7, color='lightblue', 
           s=60, label='Datos Entrenamiento', edgecolors='blue')
plt.scatter(X_test_simple, y_test_simple, alpha=0.7, color='lightcoral', 
           s=60, label='Datos Prueba', edgecolors='red')

# Línea de regresión
x_rango = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
y_linea = modelo_simple.predict(x_rango)
plt.plot(x_rango, y_linea, 'green', linewidth=3, 
         label=f'Línea de Regresión\nR² = {r2_simple:.3f}')

plt.xlabel('Longitud del Sépalo (cm)')
plt.ylabel('Longitud del Pétalo (cm)')
plt.title('Regresión Lineal Simple\nLongitud Sépalo vs Longitud Pétalo')
plt.legend()
plt.grid(True, alpha=0.3)

# Análisis de residuos
plt.subplot(1, 2, 2)
residuos = y_test_simple - y_pred_simple
plt.scatter(y_pred_simple, residuos, alpha=0.7, color='purple', s=60, edgecolors='darkviolet')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos (Error)')
plt.title('Análisis de Residuos\nModelo Simple')
plt.grid(True, alpha=0.3)

plt.suptitle('🔵 REGRESIÓN LINEAL SIMPLE', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# GRÁFICA 2: Regresión Lineal Múltiple
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Valores reales vs predichos
plt.scatter(y_test_mult, y_pred_mult, alpha=0.7, color='orange', s=70, edgecolors='darkorange')
# Línea de predicción perfecta
min_val = min(y_test_mult.min(), y_pred_mult.min())
max_val = max(y_test_mult.max(), y_pred_mult.max())
plt.plot([min_val, max_val], [min_val, max_val], 'red', linewidth=2, label='Predicción Perfecta')
plt.xlabel('Valores Reales')
plt.ylabel('Valores Predichos')
plt.title(f'Modelo Múltiple\nR² = {r2_multiple:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)

# Importancia de variables
plt.subplot(1, 2, 2)
vars_cortas = ['Long. Sépalo', 'Ancho Sépalo', 'Ancho Pétalo']
importancias = [imp for _, imp in importancia]
colores = ['skyblue', 'lightcoral', 'lightgreen']

barras = plt.bar(vars_cortas, importancias, color=colores, alpha=0.8, edgecolor='black')
plt.ylabel('Importancia (|Coeficiente|)')
plt.title('Importancia de Variables\nModelo Múltiple')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Añadir valores en las barras
for barra, imp in zip(barras, importancias):
    plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.05,
            f'{imp:.3f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('🟠 REGRESIÓN LINEAL MÚLTIPLE', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# GRÁFICA 3: Comparación de Modelos (Sépalo vs Pétalo)
plt.figure(figsize=(14, 6))

# Subplot 1: Comparación de R²
plt.subplot(1, 3, 1)
modelos = ['Modelo\nSimple', 'Modelo\nMúltiple']
r2_valores = [r2_simple, r2_multiple]
colores_comp = ['lightblue', 'orange']

barras_r2 = plt.bar(modelos, r2_valores, color=colores_comp, alpha=0.8, edgecolor='black')
plt.ylabel('R² Score')
plt.title('Comparación de Modelos\nR² (Mayor = Mejor)')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3, axis='y')

# Valores en las barras
for barra, valor in zip(barras_r2, r2_valores):
    plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.02,
            f'{valor:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Subplot 2: Regresión Sépalo vs Pétalo (vista completa)
plt.subplot(1, 3, 2)
plt.scatter(df['Longitud Sépalo'], df['Longitud Pétalo'], 
           alpha=0.7, color='steelblue', s=50, edgecolors='navy')
plt.plot(x_rango, y_linea, 'red', linewidth=3, label=f'Regresión (r = {correlacion:.3f})')
plt.xlabel('Longitud Sépalo (cm)')
plt.ylabel('Longitud Pétalo (cm)')
plt.title('Relación Sépalo vs Pétalo\n(Todos los datos)')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Comparación de errores
plt.subplot(1, 3, 3)
errores = [mse_simple, mse_multiple]
barras_error = plt.bar(modelos, errores, color=colores_comp, alpha=0.8, edgecolor='black')
plt.ylabel('Error Cuadrático Medio')
plt.title('Comparación de Errores\n(Menor = Mejor)')
plt.grid(True, alpha=0.3, axis='y')

# Valores en las barras
for barra, error in zip(barras_error, errores):
    plt.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 0.01,
            f'{error:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.suptitle('📊 COMPARACIÓN: SÉPALO vs PÉTALO', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ===============================================
# RESUMEN FINAL EN ESPAÑOL
# ===============================================
print("\n" + "="*60)
print("🎯 RESUMEN FINAL")
print("="*60)

mejora_r2 = r2_multiple - r2_simple
reduccion_error = mse_simple - mse_multiple
porcentaje_mejora = (mejora_r2 / r2_simple * 100)

print(f"\n📊 COMPARACIÓN DE RESULTADOS:")
print(f"• Modelo Simple (1 variable):    R² = {r2_simple:.3f} | Error = {mse_simple:.3f}")
print(f"• Modelo Múltiple (3 variables): R² = {r2_multiple:.3f} | Error = {mse_multiple:.3f}")

print(f"\n🚀 MEJORAS DEL MODELO MÚLTIPLE:")
print(f"• Incremento en R²: +{mejora_r2:.3f} puntos")
print(f"• Reducción en error: -{reduccion_error:.3f}")
print(f"• Mejora relativa: {porcentaje_mejora:.1f}% mejor que el modelo simple")

print(f"\n💡 CONCLUSIONES:")
print(f"• La variable más importante es: {importancia[0][0]}")
print(f"• El modelo múltiple explica {r2_multiple*100:.1f}% de la variabilidad")
print(f"• Usar múltiples variables SÍ mejora significativamente la predicción")
print(f"• La correlación Sépalo-Pétalo es muy fuerte (r = {correlacion:.3f})")

print("\n" + "="*60)