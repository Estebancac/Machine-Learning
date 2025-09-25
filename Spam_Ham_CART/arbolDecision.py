import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_dataset(file_path):
    """Cargar el dataset real"""
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset cargado exitosamente: {data.shape}")
        print(f"Columnas encontradas: {list(data.columns)}")
        
        # Verificar que existe la columna Label
        if 'Label' not in data.columns:
            raise ValueError("El dataset debe tener una columna llamada 'Label'")
            
        return data
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        print("Asegúrate de que:")
        print("1. La ruta del archivo es correcta")
        print("2. El archivo tiene las columnas correctas")
        print("3. Una de las columnas se llama 'Label'")
        return None

def run_experiment(data, test_size=0.3, random_state=None, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """Ejecutar un experimento completo con configuraciones específicas"""
    
    # 1. PREPARACIÓN DE DATOS
    X = data.drop('Label', axis=1)
    y = data['Label']
    
    # Dividir datos con configuración específica
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 2. CREAR MODELO con configuración específica
    model = DecisionTreeClassifier(
        criterion='gini',  # CART usa Gini
        random_state=random_state,
        max_depth=max_depth,  # CONFIGURACIÓN VARIABLE
        min_samples_split=min_samples_split,  # CONFIGURACIÓN VARIABLE
        min_samples_leaf=min_samples_leaf  # CONFIGURACIÓN VARIABLE
    )
    
    # 3. ENTRENAR MODELO
    model.fit(X_train, y_train)
    
    # 4. PREDICCIÓN
    y_pred = model.predict(X_test)
    
    # 5. EVALUACIÓN
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Z Score (comparado con clasificación aleatoria 50%)
    expected_acc = 0.5
    n = len(y_test)
    std_dev = np.sqrt(expected_acc * (1 - expected_acc) / n)
    z_score = (accuracy - expected_acc) / std_dev
    
    return accuracy, f1, z_score

def run_multiple_experiments(data, n_experiments=50):
    """Ejecutar múltiples experimentos con diferentes configuraciones"""
    
    print(f"Ejecutando {n_experiments} experimentos con distintas configuraciones...")
    
    results = {
        'accuracy': [],
        'f1_score': [],
        'z_score': [],
        'test_size': [],
        'max_depth': [],
        'min_samples_split': [],
        'min_samples_leaf': [],
        'experiment': []
    }
    
    # CONFIGURACIONES QUE SE VARÍAN EN CADA EXPERIMENTO:
    for i in range(n_experiments):
        # 1. VARIAR TAMAÑO DEL CONJUNTO DE TEST (20% a 40%)
        test_size = np.random.uniform(0.2, 0.4)
        
        # 2. VARIAR HIPERPARÁMETROS DEL ÁRBOL DE DECISIÓN
        max_depth = np.random.choice([None, 5, 10, 15, 20])  # Profundidad máxima
        min_samples_split = np.random.choice([2, 5, 10, 20])  # Mínimo para dividir
        min_samples_leaf = np.random.choice([1, 2, 5, 10])    # Mínimo en hojas
        
        # 3. VARIAR SEMILLA ALEATORIA
        random_state = i
        
        print(f"Experimento {i+1}: test_size={test_size:.2f}, max_depth={max_depth}, "
              f"min_split={min_samples_split}, min_leaf={min_samples_leaf}")
        
        # Ejecutar experimento con estas configuraciones
        accuracy, f1, z_score = run_experiment(
            data, test_size, random_state, max_depth, min_samples_split, min_samples_leaf
        )
        
        # Guardar resultados
        results['accuracy'].append(accuracy)
        results['f1_score'].append(f1)
        results['z_score'].append(z_score)
        results['test_size'].append(test_size)
        results['max_depth'].append(max_depth)
        results['min_samples_split'].append(min_samples_split)
        results['min_samples_leaf'].append(min_samples_leaf)
        results['experiment'].append(i+1)
        
        if (i + 1) % 10 == 0:
            print(f"✅ Completados {i + 1} experimentos")
    
    return pd.DataFrame(results)

def plot_final_tree(data, feature_names):
    """Entrenar y visualizar el árbol de decisión final"""
    
    print("\n=== VISUALIZACIÓN DEL ÁRBOL DE DECISIÓN FINAL ===")
    
    # Usar la mejor configuración encontrada o una configuración estándar
    X = data.drop('Label', axis=1)
    y = data['Label']
    
    # Dividir datos con una configuración representativa
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Crear modelo final con parámetros que permitan buena visualización
    final_model = DecisionTreeClassifier(
        criterion='gini',
        max_depth=4,  # Limitamos profundidad para mejor visualización
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    # Entrenar el modelo final
    final_model.fit(X_train, y_train)
    
    # Evaluar el modelo final
    y_pred = final_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred)
    final_f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Rendimiento del árbol final:")
    print(f"• Exactitud: {final_accuracy:.4f}")
    print(f"• F1 Score: {final_f1:.4f}")
    
    # Crear visualización del árbol
    plt.figure(figsize=(20, 12))
    plot_tree(final_model, 
              feature_names=feature_names,
              class_names=['HAM', 'SPAM'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('Árbol de Decisión Final - Clasificador SPAM/HAM', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Mostrar reglas del árbol en formato texto
    print(f"\nReglas del árbol de decisión:")
    print("=" * 50)
    tree_rules = export_text(final_model, feature_names=feature_names)
    print(tree_rules)
    
    # Mostrar importancia de características
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nImportancia de características:")
    print(feature_importance.to_string(index=False))
    
    # Graficar importancia de características
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importancia')
    plt.title('Importancia de Características en el Árbol Final')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return final_model
    """Graficar resultados de las tres métricas"""
    
def plot_results(results_df):
    """Graficar resultados de las tres métricas"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Resultados del Clasificador SPAM/HAM - Decision Tree (CART)', fontsize=14)
    
    # Exactitud
    axes[0].hist(results_df['accuracy'], bins=15, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Distribución de Exactitud')
    axes[0].set_xlabel('Exactitud')
    axes[0].set_ylabel('Frecuencia')
    axes[0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1].hist(results_df['f1_score'], bins=15, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title('Distribución de F1 Score')
    axes[1].set_xlabel('F1 Score')
    axes[1].set_ylabel('Frecuencia')
    axes[1].grid(True, alpha=0.3)
    
    # Z Score
    axes[2].hist(results_df['z_score'], bins=15, alpha=0.7, color='red', edgecolor='black')
    axes[2].set_title('Distribución de Z Score')
    axes[2].set_xlabel('Z Score')
    axes[2].set_ylabel('Frecuencia')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_results(results_df):
    """Analizar resultados y presentar conclusiones"""
    
    print("\n=== ANÁLISIS DE RESULTADOS ===")
    
    # Estadísticas
    print("\nEstadísticas de las métricas:")
    stats = results_df[['accuracy', 'f1_score', 'z_score']].describe()
    print(stats.round(4))
    
    # Promedios y desviaciones
    print(f"\nRESUMEN:")
    print(f"• Exactitud promedio: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"• F1 Score promedio: {results_df['f1_score'].mean():.4f} ± {results_df['f1_score'].std():.4f}")
    print(f"• Z Score promedio: {results_df['z_score'].mean():.4f} ± {results_df['z_score'].std():.4f}")
    
    # Análisis de configuraciones
    print(f"\nCONFIGURACIONES UTILIZADAS:")
    print(f"• Test size: {results_df['test_size'].min():.2f} - {results_df['test_size'].max():.2f}")
    print(f"• Max depth: {results_df['max_depth'].unique()}")
    print(f"• Min samples split: {results_df['min_samples_split'].unique()}")
    print(f"• Min samples leaf: {results_df['min_samples_leaf'].unique()}")
    
    # Análisis de variabilidad
    cv_accuracy = results_df['accuracy'].std() / results_df['accuracy'].mean()
    print(f"• Coeficiente de variación (Exactitud): {cv_accuracy:.4f}")
    
    # Encontrar mejores configuraciones
    best_idx = results_df['accuracy'].idxmax()
    print(f"\nMEJOR CONFIGURACIÓN (Experimento #{results_df.loc[best_idx, 'experiment']}):")
    print(f"• Exactitud: {results_df.loc[best_idx, 'accuracy']:.4f}")
    print(f"• Test size: {results_df.loc[best_idx, 'test_size']:.2f}")
    print(f"• Max depth: {results_df.loc[best_idx, 'max_depth']}")
    print(f"• Min samples split: {results_df.loc[best_idx, 'min_samples_split']}")
    print(f"• Min samples leaf: {results_df.loc[best_idx, 'min_samples_leaf']}")
    
    # Conclusiones
    print(f"\nCONCLUSIONES:")
    
    avg_accuracy = results_df['accuracy'].mean()
    if avg_accuracy > 0.8:
        print("1. El modelo tiene un rendimiento EXCELENTE")
    elif avg_accuracy > 0.7:
        print("1. El modelo tiene un rendimiento BUENO")
    else:
        print("1. El modelo tiene un rendimiento ACEPTABLE")
    
    if cv_accuracy < 0.05:
        print("2. El modelo es MUY ESTABLE entre configuraciones")
    elif cv_accuracy < 0.1:
        print("2. El modelo es ESTABLE entre configuraciones")
    else:
        print("2. El modelo muestra VARIABILIDAD entre configuraciones")
    
    avg_z = results_df['z_score'].mean()
    if avg_z > 1.96:
        print("3. Los resultados son ESTADÍSTICAMENTE SIGNIFICATIVOS")
    else:
        print("3. Los resultados NO son estadísticamente significativos")
    
    print(f"\nEXPLICACIÓN DE VARIACIONES:")
    print("• Las variaciones se deben a cambios en:")
    print("  - Tamaño del conjunto de test (20%-40%)")
    print("  - Profundidad máxima del árbol")
    print("  - Parámetros de poda (min_samples_split, min_samples_leaf)")
    print("  - Semillas aleatorias diferentes")
    print("  - Diferentes muestras de entrenamiento y prueba")

def main(dataset_path):
    """Función principal"""
    
    print("=== CLASIFICADOR SPAM/HAM CON DECISION TREE ===\n")
    
    # Cargar tu dataset real
    data = load_dataset(dataset_path)
    if data is None:
        return None
    
    print("FASE 1: Preparación de datos")
    print(f"Dataset cargado: {data.shape[0]} muestras, {data.shape[1]-1} características")
    
    # Obtener nombres de características
    feature_names = [col for col in data.columns if col != 'Label']
    
    # Mostrar distribución de clases
    label_counts = data['Label'].value_counts()
    print(f"Distribución de clases:")
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    print("\nFASE 2-5: Experimentos múltiples (modelo, entrenamiento, predicción, evaluación)")
    results = run_multiple_experiments(data, n_experiments=50)
    
    print("\nFASE 6: Visualización de resultados")
    plot_results(results)
    
    print("\nFASE 7: Análisis y conclusiones")
    analyze_results(results)
    
    print("\nFASE 8: Visualización del árbol de decisión final")
    final_model = plot_final_tree(data, feature_names)
    
    return results, final_model

# Ejecutar el programa
if __name__ == "__main__":
    # ¡¡¡CAMBIA ESTA RUTA POR LA DE TU DATASET!!!
    dataset_path = "C:\spark_datos\emails_dataset.csv"  # <-- CAMBIA AQUÍ POR LA RUTA DE TU ARCHIVO
    results = main(dataset_path)