# Examen-Final-Modelos-y-aprendizaje
Modelo que sea capaz de identificar si un paciente tiene o no cáncer

# Datos de la base
1) Número de identificación 
2) Diagnóstico (M = maligno, B = benigno) 
3-32) Se calculan diez características de valor real para cada núcleo celular: 
a) radio (media de distancias desde el centro a puntos en el perímetro) b) textura (desviación estándar de los valores de escala de grises) 
c) perímetro 
d) área 
e) suavidad (variación local en longitudes de radio) 
f) compacidad (perímetro^2 / área - 1,0) 
g) concavidad (severidad de las porciones cóncavas del contorno) 
h) puntos cóncavos (número de porciones cóncavas del contorno) 
i) simetría
 j) dimensión fractal ("aproximación de la línea costera" - 1)
# Objetivo: 
Crear mínimo 3 modelo que sea capaz de identificar si un paciente tiene o no cáncer
Decidir cuál es el mejor modelo justificando la respuesta en base a las matrices de confusión que aparecen al evaluar el error en training y en test

# Nº 1 RANDOM FOREST 
## RAZONES DEL USO DEL MODELO:
El modelo Random Forest comúnmente es utilizado en problemas de clasificación como la identificación de cáncer por varias razones:
- Alta precisión : Random Forest es conocido por su alta precisión en la clasificación, incluso en conjuntos de datos complejos y de alta dimensionalidad como el que podría surgir en el diagnóstico del cáncer, Random Forest reduce el sobreajuste y produce resultados más precisos.
- Robustez frente al sobreajuste: Los modelos de Random Forest tienden a ser menos propensos al sobreajuste en comparación con un solo árbol de decisión, especialmente en conjuntos de datos con muchas características. 
- Manejo eficiente de características: Random Forest es capaz de manejar conjuntos de datos con muchas características sin necesidad de reducción de dimensionalidad previa. 
- Tolerancia a datos desequilibrados: En problemas de diagnóstico de cáncer, es común que los datos estén desequilibrados, es decir, que haya muchas más muestras de una clase que de la otra. Random Forest puede manejar esta desigualdad de manera efectiva y producir resultados equilibrados sin requerir técnicas de reequilibrio de datos.
- Identificación de características importantes: Random Forest puede proporcionar una medida de la importancia de cada característica en la clasificación, lo que puede ayudar a los médicos y expertos en salud a comprender qué características son más relevantes para el diagnóstico del cáncer.
## PROCEDIMIENTO DEL MODELO:
- Importaciones de bibliotecas: Importamos las bibliotecas necesarias para el análisis de datos y el modelado predictivo, incluyendo herramientas como Pandas, Scikit-learn, Imbalanced-learn (para la generación de datos sintéticos en el manejo de clases desbalanceadas) y Matplotlib y Seaborn para la visualización.
- División de datos: Las características (X) y las etiquetas (y) se definen utilizando el conjunto de datos. Las características son todas las columnas excepto 'ID number' y 'Diagnosis', mientras que las etiquetas son la columna 'Diagnosis'.
- Los datos se dividen en conjuntos de entrenamiento y prueba utilizando train_test_split(). Se asigna el 80% de los datos para entrenamiento y el 20% restante para pruebas.
- Imputación y normalización de datos: Se utiliza SimpleImputer para reemplazar los valores faltantes en las características con la media de esa característica en el conjunto de entrenamiento.
- Luego, se normalizan las características utilizando StandardScaler para asegurarse de que todas estén en la misma escala.
- Ajuste de hiperparámetros con GridSearchCV: Se define un espacio de hiperparámetros para el clasificador de Random Forest, que incluye el número de estimadores, la profundidad máxima del árbol y el número mínimo de muestras requeridas para dividir un nodo interno.
- Se utiliza GridSearchCV para encontrar la combinación óptima de hiperparámetros mediante validación cruzada en el conjunto de entrenamiento.
- Balanceo de clases con SMOTE: Se aplica SMOTE (Técnica de Sobremuestreo de Minorías Sintéticas) solo en el conjunto de entrenamiento para abordar el problema de desequilibrio de clases, generando muestras sintéticas de la clase minoritaria.
- Entrenamiento del modelo y validación cruzada: Se entrena el modelo de Random Forest con los datos balanceados y normalizados del conjunto de entrenamiento, también se realiza la validación cruzada para evaluar el rendimiento del modelo en términos de precisión utilizando el conjunto de entrenamiento.
- Evaluación del modelo en el conjunto de prueba: Se realizan predicciones en el conjunto de prueba y se evalúan utilizando métricas como precisión, puntaje F1 y AUC-ROC.
“Las métricas indican que el modelo tiene una precisión del 95.45%, un puntaje F1 del 96.55% y un AUC-ROC del 97.43%, lo que sugiere un buen rendimiento en la clasificación de los casos de cáncer en el conjunto de prueba.”
- Se visualizan las matrices de confusión para el conjunto de entrenamiento y prueba.

# Nº 2 REGRESION LOGISTICA 
## RAZONES DEL USO DEL MODELO:
- Interpretación sencilla: La regresión logística produce resultados que son fácilmente interpretables. La salida está en forma de probabilidades que representan la probabilidad de que un punto de datos pertenezca a una clase en particular. Además, los coeficientes asociados a cada característica proporcionan información sobre cómo esa característica contribuye a la predicción.
- Eficiencia computacional: La regresión logística es computacionalmente eficiente y rápida de entrenar, especialmente en comparación con modelos más complejos como los árboles de decisión o las redes neuronales. Esto es útil cuando se trabaja con grandes conjuntos de datos o se necesita un modelo que se pueda entrenar rápidamente.
- Manejo natural de características binarias y numéricas: La regresión logística es capaz de manejar tanto características categóricas como numéricas sin necesidad de transformación adicional. Esto es útil cuando se trabaja con conjuntos de datos que contienen una combinación de tipos de características.
- Regularización incorporada: La regresión logística tiene la capacidad de manejar el sobreajuste a través de técnicas de regularización como la regresión L1 (Lasso) y la regresión L2 (Ridge). Estas técnicas ayudan a evitar que el modelo se ajuste demasiado a los datos de entrenamiento, lo que mejora su capacidad para generalizar a nuevos datos.
- Buena para conjuntos de datos linealmente separables: Si los datos son linealmente separables, es decir, si se pueden separar utilizando una línea recta en el espacio de características, la regresión logística puede ser muy efectiva.
- Manejo natural de probabilidades: La regresión logística produce predicciones en forma de probabilidades, lo que facilita la interpretación y la toma de decisiones basadas en umbrales de probabilidad.
## PROCEDIMIENTO DEL MODELO:
- Importaciones de bibliotecas: Importamos las bibliotecas necesarias para el análisis de datos, la construcción del modelo y la evaluación del rendimiento, incluyendo Scikit-learn, Pandas y Matplotlib.
- Definición de características y etiquetas: Se definen las características (X) como todas las columnas excepto 'ID number' y 'Diagnosis', y las etiquetas (y) como la columna 'Diagnosis' del conjunto de datos.
- División de datos en conjuntos de entrenamiento y prueba: Se utilizan train_test_split() para dividir los datos en conjuntos de entrenamiento y prueba en una proporción del 80% para entrenamiento y 20% para pruebas. Esto proporciona datos independientes para entrenar y evaluar el modelo.
- Inicialización y entrenamiento del modelo de regresión logística: Se inicializa un objeto de regresión logística utilizando LogisticRegression() y se entrena el modelo con los datos de entrenamiento utilizando fit().
- Predicciones en datos de prueba: Se realizan predicciones en los datos de prueba utilizando predict() y se almacenan en y_pred.
- Transformación de etiquetas categóricas a valores numéricos: Las etiquetas categóricas 'M' (maligno) y 'B' (benigno) se convierten en valores numéricos 1 y 0, respectivamente, para calcular métricas de evaluación.
- Cálculo de métricas de evaluación: Se calculan varias métricas de evaluación del rendimiento del modelo, incluyendo precisión, recall, puntaje F1, área bajo la curva ROC (ROC AUC) y precisión (accuracy). Estas métricas proporcionan una evaluación completa del rendimiento del modelo.
- Impresión de métricas de evaluación: Se imprimen las métricas de evaluación calculadas para evaluar el rendimiento del modelo.
“En este caso, la precisión es del 97.56%, lo que significa que el 97.56% de los casos clasificados como positivos por el modelo realmente tienen cáncer.”
- Obtención de matrices de confusión: Se obtienen matrices de confusión tanto para el conjunto de entrenamiento como para el conjunto de prueba utilizando confusion_matrix().
- Visualización de matrices de confusión: Se visualizan las matrices de confusión usando heatmap() de Seaborn para proporcionar una representación gráfica de las predicciones del modelo en comparación con las etiquetas reales.

# Nº 3 RED NEURONAL 
## RAZONES DEL USO DEL MODELO:
- Capacidad para aprender representaciones complejas: Las redes neuronales pueden aprender representaciones complejas de los datos, lo que les permite
capturar relaciones no lineales y patrones sutiles en los conjuntos de datos médicos. Esto es crucial en problemas como la detección de cáncer, donde
las características pueden estar altamente interconectadas y no linealmente separables.
- Adaptabilidad a diferentes tipos de datos: Las redes neuronales pueden manejar eficazmente una amplia variedad de tipos de datos, incluidos datos
estructurados (como valores numéricos), datos no estructurados (como imágenes médicas) y datos secuenciales (como registros de tiempo). Esto es útil
en el contexto del diagnóstico del cáncer, donde la información puede provenir de diversas fuentes y formatos.
- Flexibilidad arquitectónica: Las redes neuronales ofrecen una amplia variedad de arquitecturas y capas, lo que permite adaptar el modelo a las
características específicas del problema. Por ejemplo, en el diagnóstico del cáncer, se pueden utilizar arquitecturas como redes neuronales
convolucionales (CNN) para procesar imágenes médicas o redes neuronales recurrentes (RNN) para analizar datos secuenciales.
- Capacidad para manejar grandes volúmenes de datos: Las redes neuronales pueden manejar grandes conjuntos de datos con eficacia, lo que es importante en el diagnóstico del cáncer, donde pueden haber miles o incluso millones de pacientes con múltiples características observadas.
- Aprendizaje jerárquico y abstracción de características: Las redes neuronales pueden aprender automáticamente características jerárquicas y abstracciones de los datos, lo que les permite identificar características relevantes para la predicción del cáncer sin necesidad de una ingeniería de características manual.
- Regularización y prevención del sobreajuste: Las redes neuronales pueden aplicar técnicas de regularización como la disminución del aprendizaje, la eliminación de características y el abandono para prevenir el sobreajuste y mejorar la generalización del modelo.
## PROCEDIMIENTO DEL MODELO:
- Importación de bibliotecas: Se importan las bibliotecas necesarias, incluyendo Keras, TensorFlow, Matplotlib y Scikit-learn.
- Eliminación de columnas irrelevantes: Se elimina la columna 'ID number', ya que no es relevante para el modelo.
- Codificación de etiquetas categóricas: Se utiliza LabelEncoder para convertir la columna 'Diagnosis' en valores numéricos.
- División de datos: Se dividen los datos en características (X) y etiquetas (y), y luego en conjuntos de entrenamiento y prueba utilizando train_test_split().
- Escalado de características: Se escalan las características utilizando StandardScaler() para normalizarlas.
- Construcción del modelo de red neuronal: Se crea un modelo secuencial (Sequential) de Keras.
- Se agregan capas densas (Dense) con funciones de activación ReLU y sigmoide. La capa de entrada (input_shape) tiene el mismo número de características que los datos de entrada.
- Compilación del modelo: Se compila el modelo especificando el optimizador (adam), la función de pérdida (binary_crossentropy) y las métricas (accuracy).
- Entrenamiento del modelo: El modelo se entrena con los datos de entrenamiento utilizando fit(), especificando el número de épocas y el tamaño del lote.
- Evaluación del modelo en el conjunto de prueba: Se evalúa el modelo en el conjunto de prueba utilizando evaluate(), y se calcula la pérdida y la precisión del modelo.
“En este caso, la precisión del modelo en el conjunto de prueba es del 97.37%, lo que significa que el 97.37% de las predicciones del modelo en el conjunto de prueba son correctas”. 
- Visualización de matrices de confusión: Se calculan las matrices de confusión para el conjunto de entrenamiento y prueba utilizando confusion_matrix().
- Las matrices de confusión:  se visualizan utilizando heatmap() de Seaborn para proporcionar una representación gráfica de las predicciones del modelo en comparación con las etiquetas reales.

# Nº 4 DECISIONTREECLASSIFIER
## RAZONES DEL USO DEL MODELO:
- Interpretabilidad: Los árboles de decisión son modelos altamente interpretables. Esto significa que se pueden visualizar y comprender fácilmente, lo que permite a los profesionales de la salud y a los investigadores entender cómo se toman las decisiones de clasificación.
Manejo de datos mixtos: Los árboles de decisión pueden manejar eficazmente conjuntos de datos que contienen tanto variables numéricas como categóricas. Esto es útil en el contexto del diagnóstico del cáncer, donde pueden haber diferentes tipos de características médicas.
- No requiere normalización de datos: A diferencia de algunos otros algoritmos, como las redes neuronales, los árboles de decisión no requieren que los datos estén normalizados o estandarizados. Esto simplifica el proceso de preparación de datos y hace que el modelo sea más fácil de implementar.
- Manejo natural de características irrelevantes: Los árboles de decisión tienden a ignorar las características irrelevantes en el proceso de construcción del árbol. Esto significa que pueden manejar conjuntos de datos con características redundantes o irrelevantes sin necesidad de preprocesamiento adicional.
- Capacidad de manejar conjuntos de datos desequilibrados: Los árboles de decisión pueden ser robustos frente a conjuntos de datos desequilibrados, donde una clase puede ser significativamente más frecuente que otra. Esto es útil en el contexto del diagnóstico del cáncer, donde las muestras de cáncer pueden ser mucho menos frecuentes que las muestras no cancerosas.
- Buena generalización: Aunque los árboles de decisión tienden a ser propensos al sobreajuste, se pueden aplicar técnicas como la poda del árbol, la limitación de la profundidad del árbol y la especificación de un número mínimo de muestras por hoja para mejorar la generalización del modelo.
## PROCEDIMIENTO DEL MODELO:
- Importación de bibliotecas: Se importan las bibliotecas necesarias, incluyendo train_test_split para dividir los datos, DecisionTreeClassifier para el modelo de árbol de decisión, accuracy_score para calcular la precisión del modelo, classification_report para obtener un informe detallado de la clasificación, confusion_matrix para calcular las matrices de confusión y seaborn y matplotlib para visualizar las matrices de confusión.
- División de datos: Se dividen los datos en conjuntos de entrenamiento y prueba utilizando train_test_split(). El conjunto de prueba es el 20% de los datos totales, y se establece una semilla (random_state) para garantizar que la división sea reproducible.
- Inicialización del modelo de árbol de decisión: Se inicializa un modelo de árbol de decisión utilizando DecisionTreeClassifier().
- Entrenamiento del modelo: Se entrena el modelo utilizando los datos de entrenamiento (X_train y y_train) mediante el método fit().
- Predicciones en el conjunto de prueba: Se realizan predicciones en el conjunto de prueba (X_test) utilizando el método predict().
- Evaluación del rendimiento del modelo: Se calcula la precisión del modelo comparando las etiquetas verdaderas (y_test) con las predicciones utilizando accuracy_score().
“Es el valor de la precisión del modelo, que en este caso es aproximadamente 93.86%. Esto significa que alrededor del 93.86% de las muestras en el conjunto de prueba fueron clasificadas correctamente por el modelo de árbol de decisión. En otras palabras, el modelo tiene una tasa de predicciones correctas del 93.86%.”
- Reporte de clasificación: Se genera un reporte detallado de la clasificación utilizando classification_report(). Este reporte proporciona métricas como precisión, recuperación (recall), puntaje F1 y soporte para cada clase.
- Matrices de confusión: Se calculan las matrices de confusión para los conjuntos de entrenamiento y prueba utilizando confusion_matrix().

# Conclusiones:
Todos los modelos muestran un buen rendimiento en términos de precisión, pero cada uno tiene sus propias fortalezas y debilidades. La elección del modelo más adecuado depende mucho de más condiciones como la importancia de minimizar los falsos positivos versus los falsos negativos en el contexto clínico específico, así como la capacidad de interpretación del modelo y la eficiencia computacional. Pero el mejor modelo justificando la respuesta en base a las matrices de confusión que aparecen al evaluar el error en training y en test.


•	Red Neuronal también tuvo un número bajo de falsos negativos (FN), lo que significa que el modelo también es bueno y logra identifican a los pacientes con cáncer correctamente, su presión del modelo también fue alto con 97.37%

•	Regresión Logística tiene un rendimiento similar a Random Forest y Red Neuronal aunque tuvo un poco más de falso negativos, pero su presión fue muy buena también con 97.56%  

•	DecisionTreeClassifier tiene el peor desempeño en términos de falsos negativos, lo que significa que identifica incorrectamente más casos de cáncer como negativos con una precisión del 93.86%

•	El modelo Random Forest se destaca como la mejor opción debido a varios factores:

- Menor cantidad de falsos negativos: En un problema de detección de cáncer, los falsos negativos (clasificar incorrectamente a alguien como no portador de cáncer cuando en realidad lo tienen) son especialmente preocupantes, ya que podrían llevar a la falta de tratamiento oportuno para los pacientes que realmente lo necesitan. Random Forest tuvo solo 1 falso negativo, lo que indica una capacidad alta para identificar correctamente los casos de cáncer.
- Alta precisión: Aunque tanto Random Forest como Regresión Logística tienen altas precisiones, Random Forest logra una precisión del 95.45%, lo que significa que el 95.45% de las predicciones son correctas. Esta alta precisión demuestra la capacidad del modelo para realizar clasificaciones precisas en los datos de prueba.
- Menor riesgo de sobreajuste: Random Forest es conocido por su robustez frente al sobreajuste. Esto significa que es menos probable que el modelo se ajuste demasiado a los datos de entrenamiento y tenga un rendimiento deficiente en datos nuevos. Su capacidad para reducir el sobreajuste es crucial para garantizar que el modelo generalice bien a datos no vistos.
En resumen, Random Forest ofrece un equilibrio entre precisión, capacidad para manejar características complejas, robustez frente al sobreajuste y capacidad para manejar datos desequilibrados. Por lo tanto, basándonos en las métricas y considerando la naturaleza del problema de detección de cáncer, el modelo Random Forest es la mejor opción para este caso.





