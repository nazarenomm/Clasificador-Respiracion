# Detección de Broncoespasmos para una plataforma mHealth

## Descarga y Organización del Dataset ICBHI

Para reproducir el pipeline es necesario descargar el dataset **ICBHI Respiratory Sound Database**.

### 1. Descargar el dataset

El dataset puede obtenerse desde:

* **Sitio oficial del ICBHI 2017 Challenge:** [https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)

### 2. Organización del dataset

Una vez descargado, colocarlo dentro de la carpeta `dataset/`, respetando la siguiente estructura exacta:

```
dataset/
└── icbhi/
    ├── ICBHI_Final_Database/
    │    └── notaciones/                  # Carpeta con anotaciones
    │        ├── demographic.txt
    │        ├── diagnosis.txt
    │        ├── filename_differences.txt
    │        └── filename_format.txt
    └── events/                           # Carpeta con eventos
```

---

## Descomprimir datasets

Es necesario descomprimir el dataset de ruido extraído del **Microsoft Scalable Noisy Speech Dataset** ([https://www.kaggle.com/datasets/abdelrhamanfakhry/noise-data-set](https://www.kaggle.com/datasets/abdelrhamanfakhry/noise-data-set)). Se hallan en la carpeta dataset, descomprimirlos en esa misma ruta.

---

## Scripts a ejecutar (Preprocesado)

El preprocesamiento de los audios originales está dividido en distintos _scripts_. A continuación se detalla qué archivo se encarga de cada parte:

### 1. Segmentación en ciclos

**Script:** `data_preprocesada/preprocesado_ciclos.py`  
Genera los ciclos respiratorios a partir de los audios originales.

### 2. Segmentación en ventanas

**Script:** `data_preprocesada/preprocesado_ventanas.py`
Divide los audios en ventanas temporales a partir de los audios originales.

### 3. Segmentación de ruido

**Script:** `dataset/ruido_largo/segmentar_ruido.py`  
Se utiliza para segmentar muestras largas de ruido en fragmentos pequeños utilizados para _augmentations_.

---

## Notebooks

**Procesado:** `data_procesada/procesado_ciclos.ipynb` o `data_procesada/procesado_ventanas.ipynb`
Esta notebook genera los mel-espectrogramas y matrices finales que se utilizarán para el entrenamiento de modelos.

En la notebook, cambiar la variable `dataset` según corresponda. Los valores utilizados hasta ahora incluyen:

* `'crudos'`
* `'aug'`
* `'crudos_cuadrados'` (sólo ventanas temporales)
* `'aug_cuadrados'` (sólo ventanas temporales)

Una vez ejecutado el notebook se creará una carpeta `data_procesada/ciclos/dataset` o `data_procesada/ventanas/dataset` con los conjuntos necesarios para entrenar los modelos.

**Entrenamiento:** `modelos_clasicos/entrenamiento_rf.ipynb` o `neural_network/entrenamiento_mobile_net.ipynb`

En la notebook, cambiar la variable `dataset` según corresponda. Los valores utilizados hasta ahora incluyen:

+ `'ciclos/crudos'`
+ `'ciclos/aug'`
+ `'ventanas/crudos'`
+ `'ventanas/aug'`
+ `'ventanas/crudos_cuadrados'` (sólo ventanas temporales)
+ `'ventanas/aug_cuadrados'` (sólo ventanas temporales)

---

## Ejecución de notebooks desde el path base

Las notebooks están diseñadas para ejecutarse desde el **directorio base del proyecto**.

Si se trabaja desde VS-Code, asegurarse de agregar al archivo `.vscode/settings.json`:
```json
{
    "jupyter.notebookFileRoot": "${workspaceFolder}",
}
```

Esto permite que los paths relativos se resuelvan correctamente.

---

## Mapa del repositorio

Ejemplo de estructura recomendada:

```
dataset/
└── icbhi/
│   ├── ICBHI_Final_Database/
│   │   ├── notaciones/
│   │   └── events/
│   ├── audios_propios/ (descomprimir)
│   └── ruido_largo/ (descomprimir)
│       └── segmentar_ruido.py
│
├── data_preprocesada/
│   ├── preprocesado_ciclos.py
│   └── preprocesado_ventanas.py
│
├── data_procesada/
│   ├── procesado_ciclos.ipynb
│   └── procesado_ventanas.ipynb
│
├── modelos_clasicos/
│   ├── modelos/
│   └── entrenamiento_rf.ipynb
│
├── neural_network/
│   ├── modelos/
│   └── entrenamiento_mobile_net.ipynb
│
├── transformers/
│   └── (transformers para el procesamiento de audios)
│
├── segmentador/
│   └── (implementación rudimentaria de un segmentador de ciclos respiratorios)
│
├── utils/
│   └── (funciones auxiliares)
│
├── informe/
│   └── (informe del proyecto)
│
├── config.py
│
├── evaluacion_modelos.ipynb
│
├── requirements.txt
│
└── README.md
```