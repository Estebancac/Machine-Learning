# 📧 Clasificador SPAM/HAM - Decision Tree (CART)

Sistema de clasificación de correos electrónicos utilizando árboles de decisión con algoritmo CART.

## 🎯 Objetivo
Desarrollar y evaluar un clasificador SPAM/HAM usando Decision Tree con 50 experimentos independientes para medir robustez y rendimiento.

## 🏆 Resultados Obtenidos

### Métricas de Rendimiento
| Métrica | Resultado | Interpretación |
|---------|-----------|----------------|
| **Exactitud** | 100% (50/50 experimentos) | Clasificación perfecta |
| **F1 Score** | 100% | Balance perfecto precisión/recall |
| **Z Score** | 17.32 ± 1.81 | Altamente significativo (p < 0.0001) |
| **Coeficiente de Variación** | 0.0000 | Estabilidad máxima |

### 🌳 Estructura del Árbol Final
El modelo encontró una regla de clasificación simple y efectiva:
```
Sender_Reputation ≤ 4.5 → SPAM (347 muestras)
Sender_Reputation > 4.5 → HAM (353 muestras)
```

### 📊 Análisis de Robustez
- **50 experimentos** con configuraciones diferentes
- **Invarianza total** a cambios de hiperparámetros
- **Rendimiento consistente** en divisiones train/test del 60/40 al 80/20
- **Separabilidad perfecta** usando una sola característica

## 📈 Dataset Utilizado
- **Tamaño:** 1,000 muestras
- **Distribución:** Balanceada (502 SPAM, 498 HAM)
- **Características:** 9 variables (Email_Length, Num_Links, Sender_Reputation, etc.)


## 📝 Conclusiones
1. **Rendimiento excepcional:** 100% exactitud en todos los experimentos
2. **Simplicidad del modelo:** Una sola característica suficiente para clasificación perfecta
3. **Alta interpretabilidad:** Regla de decisión clara y explicable
4. **Robustez comprobada:** Resultados consistentes bajo múltiples configuraciones

## 📊 Archivos Generados
- `Resultados.png` - Histogramas de métricas
- `ArbolDecision.png` - Visualización del árbol
- `Informe_CART.pdf` - Análisis completo

---
**Autor:** Esteban Guzman y Karol Diaz | **Curso:** Machine Learning | **Año:** 2025
