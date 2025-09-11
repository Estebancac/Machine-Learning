import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report, 
                           precision_recall_fscore_support, roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

print("🔍 ANÁLISIS COMPLETO DEL MODELO DE CLASIFICACIÓN DE EMAILS")
print("="*60)

# 1. Cargar y explorar el dataset
print("\n📁 CARGANDO Y EXPLORANDO EL DATASET")
print("-"*40)

dataset = pd.read_csv("C:/spark_datos/emails_dataset.csv")

print(f"📊 Forma del dataset: {dataset.shape}")
print(f"🏷️ Columnas disponibles: {list(dataset.columns)}")
print(f"❓ Valores nulos por columna:\n{dataset.isnull().sum().sum()} valores nulos en total")

# 2. Análisis de la variable objetivo
print("\n🎯 ANÁLISIS DE LA VARIABLE OBJETIVO")
print("-"*40)

print("Distribución de clases:")
class_distribution = dataset["Label"].value_counts()
class_proportion = dataset["Label"].value_counts(normalize=True)

for label, count in class_distribution.items():
    proportion = class_proportion[label]
    print(f"  {label}: {count} samples ({proportion:.2%})")

# Verificar si está balanceado
if min(class_proportion) < 0.1:
    print("⚠️  ADVERTENCIA: Dataset muy desbalanceado!")
elif min(class_proportion) < 0.3:
    print("⚡ Dataset moderadamente desbalanceado")
else:
    print("✅ Dataset relativamente balanceado")

# 3. Preparación de datos
print("\n🔧 PREPARACIÓN DE DATOS")
print("-"*40)

X = dataset.drop("Label", axis=1)
y = dataset["Label"]

print(f"Features: {X.shape[1]} columnas")
print(f"Samples: {X.shape[0]} filas")

# Verificar tipos de datos
print(f"Tipos de datos en features:")
print(X.dtypes.value_counts())

# 4. Evaluación con múltiples divisiones train/test
print("\n🎲 EVALUACIÓN CON MÚLTIPLES RANDOM STATES")
print("-"*40)

results_multiple_splits = []
random_states = [0, 42, 123, 456, 789]

for rs in random_states:
    # División train/test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=rs, stratify=y
    )
    
    # Escalado
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    # Modelo
    model = LogisticRegression(max_iter=200, random_state=rs, C=1.0)
    model.fit(x_train_scaled, y_train)
    
    # Predicción
    pred = model.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, pred)
    results_multiple_splits.append(accuracy)
    
    print(f"Random State {rs}: {accuracy:.4f} ({accuracy*100:.2f}%)")

mean_acc = np.mean(results_multiple_splits)
std_acc = np.std(results_multiple_splits)
print(f"\n📈 Accuracy promedio: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"   Rango: {min(results_multiple_splits):.4f} - {max(results_multiple_splits):.4f}")

# 5. Validación cruzada estratificada
print("\n🔄 VALIDACIÓN CRUZADA ESTRATIFICADA")
print("-"*40)

# Preparar datos completos
scaler_full = StandardScaler()
X_scaled = scaler_full.fit_transform(X)

# Validación cruzada con diferentes métricas
cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_cv = LogisticRegression(max_iter=200, random_state=42, C=1.0)

# Múltiples métricas
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = {}

for metric in scoring_metrics:
    scores = cross_val_score(model_cv, X_scaled, y, cv=cv_folds, scoring=metric)
    cv_results[metric] = scores
    print(f"{metric.upper():>12}: {scores.mean():.4f} ± {scores.std():.4f}")

# 6. Análisis detallado del mejor modelo
print("\n🔬 ANÁLISIS DETALLADO DEL MODELO")
print("-"*40)

# Entrenar modelo final
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

final_model = LogisticRegression(max_iter=200, random_state=42, C=1.0)
final_model.fit(x_train_scaled, y_train)

# Predicciones
y_pred = final_model.predict(x_test_scaled)
y_pred_proba = final_model.predict_proba(x_test_scaled)[:, 1]

# Métricas detalladas
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC:  {auc_score:.4f}")

# Matriz de confusión
print("\n📊 MATRIZ DE CONFUSIÓN:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Reporte por clase
print("\n📋 REPORTE DETALLADO POR CLASE:")
unique_labels = sorted(y.unique())
target_names = [f"Clase_{label}" for label in unique_labels]
print(classification_report(y_test, y_pred, target_names=target_names))

# 7. Análisis de consistencia entre métricas de entrenamiento y test
print("\n⚖️  ANÁLISIS DE OVERFITTING/UNDERFITTING")
print("-"*40)

# Accuracy en entrenamiento
y_train_pred = final_model.predict(x_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy Entrenamiento: {train_accuracy:.4f}")
print(f"Accuracy Test:         {test_accuracy:.4f}")
print(f"Diferencia:            {abs(train_accuracy - test_accuracy):.4f}")

if abs(train_accuracy - test_accuracy) > 0.05:
    print("⚠️  Posible overfitting detectado")
elif test_accuracy > train_accuracy:
    print("🤔 Test accuracy > Train accuracy (inusual)")
else:
    print("✅ Diferencia aceptable entre train y test")

# 8. Verificación de data leakage
print("\n🔍 VERIFICACIÓN DE DATA LEAKAGE")
print("-"*40)

print("Primeras 5 filas del dataset:")
print(dataset.head())

print(f"\nDescripción estadística de las features:")
print(X.describe())

# Correlación con la variable objetivo (si es numérica)
if pd.api.types.is_numeric_dtype(y):
    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    print(f"\nTop 5 features más correlacionadas con el target:")
    print(correlations.head())
    
    # Advertencia sobre correlaciones muy altas
    high_corr = correlations[correlations > 0.95]
    if len(high_corr) > 0:
        print(f"⚠️  ADVERTENCIA: Features con correlación > 0.95:")
        for feature, corr in high_corr.items():
            print(f"  {feature}: {corr:.4f}")

# 9. Recomendaciones finales
print("\n💡 RECOMENDACIONES FINALES")
print("-"*40)

if mean_acc > 0.95:
    print("📈 Accuracy muy alta (>95%):")
    print("  ✓ Verifica que no hay data leakage")
    print("  ✓ Confirma que el dataset es realista")
    print("  ✓ Prueba con un dataset independiente")

if std_acc > 0.05:
    print("📊 Alta variabilidad en resultados:")
    print("  ✓ Considera aumentar el tamaño del dataset")
    print("  ✓ Prueba técnicas de ensemble")

if min(class_proportion) < 0.1:
    print("⚖️  Dataset desbalanceado:")
    print("  ✓ Considera usar SMOTE o class_weight='balanced'")
    print("  ✓ Evalúa con métricas como F1-score y AUC")

print("\n🎉 ANÁLISIS COMPLETADO")
print("="*60)