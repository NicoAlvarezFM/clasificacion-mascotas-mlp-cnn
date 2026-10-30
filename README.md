# Clasificación de mascotas con redes neuronales: MLP vs CNN

Clasificador de imágenes de cuatro razas del dataset **Oxford-IIIT Pet** (samoyedo, bombay, siamés y yorkshire terrier), resuelto en dos fases con los mismos datos y la misma partición: un **Perceptrón Multicapa (MLP)** como línea base y una **Red Neuronal Convolucional (CNN)** comparada de forma objetiva con él.

**Asignatura:** TLY1102 – Técnicas Avanzadas de Machine Learning I · **Grupo 9**
**Integrantes:** Bastián Bravo - Nicolás Álvarez **
**Framework:** TensorFlow / Keras · **Semilla:** 42

---

## Resumen de resultados

| | MLP (fase 1) | **CNN C4fr (fase 2)** | Transfer learning (extensión) |
|---|---|---|---|
| Accuracy en test | 64,17 % (77/120) | **87,50 %** (105/120) | 100 % (120/120) |
| Recall mínimo por raza | 0,47 | **0,80** | 1,00 |
| Pérdida en test | 0,880 | 0,522 | – |
| Parámetros | 1.583.460 | 652.324 | 2.340.228 (82.244 entrenables) |
| Errores con confianza ≥ 90 % (sobre el total de errores) | 2 de 43 | 10 de 15 | 0 |
| ¿Cumple los KPIs? | No | **Sí** | Sí |

**KPIs del proyecto:** accuracy en test ≥ 75 % y recall mínimo por raza ≥ 0,65.

- La CNN supera al MLP en 23,3 puntos de accuracy. De las imágenes en que los modelos difieren, la CNN acierta 32 que el MLP fallaba y falla 4 que el MLP acertaba (McNemar exacto, p < 0,0001).
- El transfer learning (MobileNetV2 preentrenada con ImageNet) **no es una comparación pareja** con una CNN entrenada desde cero, porque ImageNet incluye clases de varias de estas razas. Se presenta como extensión y referencia; **el modelo final del trabajo es C4fr**.

### Desempeño por raza (test, 30 imágenes por raza)

| Raza | Recall MLP | Recall CNN | F1 MLP | F1 CNN |
|---|---|---|---|---|
| samoyed | 0,73 | 0,80 | 0,76 | 0,89 |
| bombay | 0,90 | 1,00 | 0,82 | 0,95 |
| siamese | 0,47 | 0,80 | 0,47 | 0,83 |
| yorkshire_terrier | 0,47 | 0,90 | 0,50 | 0,83 |

---

## Dataset

- **Oxford-IIIT Pet Dataset**, subconjunto de 4 razas: 800 imágenes, balanceadas (200 por raza).
- División estratificada **70 / 15 / 15** (560 train, 120 validación, 120 test) con semilla 42. El MLP y la CNN usan exactamente la misma partición.
- Preprocesamiento: redimensionado a 64×64 (interpolación bilineal), escala [0, 1] y estandarización con la media y desviación **solo del conjunto de entrenamiento** (sin fuga de información).
- **El dataset no se incluye en este repositorio.** Descárgalo desde el [sitio oficial](https://www.robots.ox.ac.uk/~vgg/data/pets/) y revisa allí su licencia.

---

## Metodología

### Fase 1 – MLP
Cinco configuraciones comparadas solo con validación. Modelo final: arquitectura piramidal 128-64-32, Dropout 0,3, Early Stopping. Alcanza 64,17 % de accuracy y se guía sobre todo por el color y el fondo de la foto, no por la forma del animal (32 de sus 43 errores cruzan especies).

### Fase 2 – CNN
Nueve configuraciones comparadas **solo con validación** (el test no interviene en ninguna decisión), más una repetición con 4 semillas de las tres candidatas finales:

| ID | Configuración | `val_accuracy` (semilla 42) |
|---|---|---|
| C0 | Arquitectura de referencia: 2 conv, Flatten, sin Dropout ni BatchNorm | 74,2 % |
| C0b | C0 + Dropout 0,3 | 81,7 % |
| C1 / C2 | 3 bloques, sin / con aumento de datos | 66,7 % / 77,5 % |
| C3 | 2 bloques, con aumento | 69,2 % |
| C4 | 4 bloques, GlobalAveragePooling | 80,8 % |
| C4f | C4 con Flatten | 84,2 % |
| C4r | C4 + `ReduceLROnPlateau` | 80,8 % |
| **C4fr** | **C4 con Flatten + `ReduceLROnPlateau`** | **89,2 %** |

Con 4 semillas, C4fr obtiene 86,9 % ± 1,9 de accuracy de validación, frente a 79,2 % ± 1,8 de C0b y 69,8 % ± 16,7 de C4 (inestable). Lo que más influyó: el aumento de datos, el Dropout, `ReduceLROnPlateau` (estabiliza la validación) y `Flatten`.

**Arquitectura final (C4fr), entrada 64×64×3:**

```
Aumento de datos (giro, rotación ±18°, zoom ±10 %) -- solo al entrenar
4 bloques: Conv2D(32/64/128/256, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
Flatten (4×4×256) → Dropout(0,3) → Dense(64, ReLU) → Dropout(0,3) → Dense(4, softmax)
```

**Entrenamiento:** Adam (0,001), entropía cruzada categórica, batch de 32, hasta 100 épocas, Early Stopping sobre `val_loss` (paciencia 15) y `ReduceLROnPlateau` (factor 0,5, paciencia 5).

### Análisis e interpretabilidad
- Matriz de confusión, precision, recall y F1 por raza.
- **Mapas de características** de la CNN y **Grad-CAM** de los errores.
- Los 15 errores de la CNN, numerados y contrastados con los del MLP (11 comunes, 4 nuevos).

### Extras
- **Latencia** (medida solo en CPU): ~10 s por época y ~39 ms por imagen en el pipeline completo (archivo → predicción).
- **Resolución 96×96:** misma accuracy de validación (88,3 %) con ~50 % más parámetros y un entrenamiento ~3,8 veces más lento; se mantiene 64×64.
- **Transfer learning** con MobileNetV2 congelada (100 % en test, ver advertencia arriba).
- **Fotos propias de cámara:** C4fr acierta 51 de 62 (82 %), por debajo de su desempeño en test.

---

## Hallazgos principales

- La CNN conserva la estructura espacial y rinde mejor con menos de la mitad de los parámetros del MLP. La mejora se concentra donde el MLP era más débil (siamese y yorkshire_terrier).
- **Limitación persistente:** la CNN sigue usando el fondo y el encuadre como atajo (por ejemplo, samoyedos sobre nieve, pasto o interiores cargados fallan en ambos modelos).
- **La CNN se equivoca con mucha seguridad** (10 de 15 errores con confianza ≥ 90 %), por lo que la probabilidad de la softmax no basta por sí sola para detectar sus errores.
- Con 120 imágenes de test, cada imagen vale 0,83 puntos de accuracy: diferencias de una o dos imágenes no son concluyentes.

---

## Estructura del repositorio

```
.
├── notebooks/
│   └── [nombre_del_notebook].ipynb   # notebook completo (fases 1 y 2)
├── models/                           # modelos y preprocesamiento guardados
│   ├── mlp_mascotas.keras
│   ├── preprocesamiento_mlp_mascotas.json
│   ├── cnn_mascotas.keras
│   ├── preprocesamiento_cnn_mascotas.json
│   └── transfer_mobilenetv2_mascotas.keras   # si se ejecutó la sección 24
├── images/                           # figuras generadas por el notebook
├── data/                             # (no versionado) dataset Oxford-IIIT Pet
└── registro_camara/                  # (no versionado) fotos propias y registros
```

---

## Cómo ejecutarlo

### Opción A: Google Colab (la usada en el desarrollo)
1. Sube el notebook a Colab.
2. Sube el ZIP del dataset (`The Oxford-IIIT Pet Dataset.zip`) a la carpeta `/content`, o deja que la celda de carga te pida seleccionarlo.
3. Ejecuta **Entorno → Reiniciar y ejecutar todo**.

### Opción B: local
1. Clona el repositorio y crea un entorno virtual.
2. Instala las dependencias:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn pillow scikit-learn scipy
   ```
3. Copia el ZIP del dataset (o la carpeta `images` ya descomprimida) en `data/`. Deja **un solo ZIP** en esa carpeta.
4. Abre el notebook desde `notebooks/` (o desde la raíz del proyecto) y ejecuta todas las celdas.

**Versiones usadas en la ejecución registrada:** Python 3.13, TensorFlow 2.21, NumPy 2.1.

### Consideraciones
- **Tiempos:** el entrenamiento de C4fr toma ~10 min en CPU. El experimento opcional de 96×96 tarda unos 36 min; puedes omitirlo.
- **Reproducibilidad:** todo usa la semilla 42, pero en GPU algunas operaciones no son deterministas, por lo que las cifras pueden variar levemente al reejecutar.
- **Celdas opcionales:** las secciones de cámara y la app de Gradio dependen de la carpeta `registro_camara/` con fotos propias; si no existe, omítelas.

---

## Limitaciones y trabajo futuro

- Ajuste fino (*fine-tuning*) de la red preentrenada y evaluación con fotos propias.
- Localizar y recortar al animal con las anotaciones del dataset para quitar el fondo.
- Conservar la proporción de la imagen en vez de deformarla a un cuadrado.
- Aumento de datos más variado y más fotos por raza, con evaluación en fotos distintas a las de entrenamiento.
- Validación cruzada y más semillas para estimar el desempeño con menos incertidumbre que un único test de 120 imágenes.
- Calibrar la confianza del modelo o agregar un umbral de "no estoy seguro".

---

## Licencia y créditos

- Dataset: *Oxford-IIIT Pet Dataset* (Parkhi et al., 2012). 
- Proyecto desarrollado para la asignatura TLY1102, Grupo 9.
