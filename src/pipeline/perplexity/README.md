# Pipeline de Cálculo de Perplexity

Este módulo calcula métricas de perplexity para todos los documentos del corpus utilizando modelos de lenguaje.

Actualmente se calculan dos métricas independientes:

- `coreguapa_perplexity`, utilizando el modelo `guaran-ia/coreguapa-lm`.
- `tweets_perplexity`, utilizando el modelo `guaran-ia/gntweets-lm`.

Las métricas calculadas se almacenan directamente en cada documento JSONL.

Por ejemplo:

```json
{
    "text": "...",
    "coreguapa_perplexity": 19.43,
    "tweets_perplexity": 137.42
}
```

El pipeline permite calcular cada métrica de forma independiente. Sin embargo, el estado esperado del corpus es que **todos los documentos contengan ambas métricas de perplexity**.

---

# Ubicación

```text
src/pipeline/perplexity
```

---

# Configuración

Antes de ejecutar el pipeline deben configurarse las siguientes variables de entorno.

```bash
export PERPLEXITY_INPUT_DIR=data/processed

export HF_HOME=.cache/huggingface
export HF_HUB_CACHE=.cache/huggingface/hub
export HF_LOCAL_FILES_ONLY=0

export PERPLEXITY_MAX_LENGTH=8192
export PERPLEXITY_STRIDE=4096
export PERPLEXITY_TEXT_CHUNK_SIZE=32768

export BATCH_SIZE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

# Variables de entorno

| Variable | Descripción |
|----------|-------------|
| `PERPLEXITY_INPUT_DIR` | Directorio que contiene los archivos JSONL procesados. |
| `HF_HOME` | Directorio utilizado por Hugging Face para almacenar los modelos descargados. |
| `HF_HUB_CACHE` | Directorio del caché de Hugging Face Hub. |
| `HF_LOCAL_FILES_ONLY` | Si vale `1`, únicamente se utilizarán modelos disponibles localmente. |
| `PERPLEXITY_MAX_LENGTH` | Cantidad máxima de tokens procesados en cada ventana deslizante. |
| `PERPLEXITY_STRIDE` | Cantidad de tokens descartados antes de construir la siguiente ventana. Debe cumplirse `0 < PERPLEXITY_STRIDE < PERPLEXITY_MAX_LENGTH`. |
| `PERPLEXITY_TEXT_CHUNK_SIZE` | Cantidad aproximada de caracteres leídos y tokenizados en cada fragmento de texto. |
| `BATCH_SIZE` | Cantidad de registros acumulados antes de escribir nuevamente los documentos procesados en el archivo JSONL. |
| `PYTORCH_CUDA_ALLOC_CONF` | Configuración del asignador de memoria CUDA utilizada para reducir la fragmentación de memoria. |

---

# Almacenamiento de las métricas

Cada documento del corpus almacena las dos métricas calculadas:

- `coreguapa_perplexity`
- `tweets_perplexity`

Por ejemplo:

```json
{
    "text": "...",
    "coreguapa_perplexity": 21.53,
    "tweets_perplexity": 115.82
}
```

Antes de calcular una métrica, el pipeline verifica si ésta ya existe en el documento.

Si la métrica solicitada ya fue calculada, no vuelve a procesarse.

Esto permite:

- reanudar ejecuciones interrumpidas;
- evitar recalcular documentos previamente procesados;
- calcular únicamente las métricas faltantes.

Aunque cada métrica puede calcularse por separado, el estado esperado del corpus es que **todos los documentos contengan ambas métricas de perplexity**.

---

# Procesamiento de documentos largos

Los modelos de lenguaje poseen una longitud máxima de contexto.

Por ejemplo:

```text
8192 tokens
```

Los documentos cuyo tamaño supera ese límite no pueden procesarse en una única inferencia.

Para evitar truncar el texto, el pipeline combina dos estrategias:

1. tokenización incremental;
2. ventanas deslizantes (*sliding windows*).

Con la siguiente configuración:

```bash
PERPLEXITY_MAX_LENGTH=8192
PERPLEXITY_STRIDE=4096
PERPLEXITY_TEXT_CHUNK_SIZE=32768
```

el flujo general es el siguiente:

1. Leer un documento desde el archivo JSONL.
2. Procesar el texto de manera incremental en fragmentos de caracteres.
3. Tokenizar únicamente el fragmento actual.
4. Agregar progresivamente los tokens obtenidos a un buffer.
5. Cuando el buffer alcanza `PERPLEXITY_MAX_LENGTH` tokens, ejecutar una inferencia utilizando esa ventana.
6. Descartar `PERPLEXITY_STRIDE` tokens del comienzo del buffer y conservar los restantes para construir la siguiente ventana.
7. Continuar leyendo y tokenizando el resto del documento hasta procesarlo completamente.

---

# Tokenización incremental

El pipeline utiliza tokenización incremental para evitar construir en memoria la secuencia completa de tokens de documentos muy grandes.

En lugar de tokenizar el documento completo en una única operación, el texto se divide en fragmentos de caracteres que son procesados de forma secuencial. Cada fragmento se tokeniza de manera independiente y los tokens generados se incorporan progresivamente a un buffer utilizado para construir las ventanas de inferencia.

Este enfoque hace que el consumo de memoria asociado a la tokenización dependa principalmente del tamaño del fragmento configurado mediante `PERPLEXITY_TEXT_CHUNK_SIZE`, en lugar de depender del tamaño total del documento.

El documento original permanece cargado en memoria mientras se procesa el registro JSONL. Lo que se evita es mantener simultáneamente una única secuencia completa de tokens correspondiente a todo el documento.

---

# Ventanas deslizantes

Una vez que los tokens son incorporados al buffer, el cálculo de la perplexity se realiza utilizando ventanas deslizantes (*sliding windows*).

Cada ventana contiene como máximo `PERPLEXITY_MAX_LENGTH` tokens. Después de procesar una ventana, se descartan `PERPLEXITY_STRIDE` tokens del comienzo del buffer, mientras que los tokens restantes se conservan para construir la siguiente ventana.

Cada ventana se evalúa de forma independiente siguiendo la receta publicada en los model cards de `coreguapa-lm` y `gntweets-lm`.


Debido al solapamiento entre ventanas, los tokens compartidos participan nuevamente en el cálculo de la pérdida de la ventana siguiente.

Este procedimiento se repite hasta procesar completamente el documento, permitiendo evaluar textos cuya longitud excede el tamaño máximo de contexto soportado por el modelo sin necesidad de truncarlos.

---

> **Nota**
>
> El algoritmo utilizado para el cálculo de la perplexity replica la receta de ventanas deslizantes publicada en los model cards de los modelos:
>
> - CoreGuapa LM: https://huggingface.co/guaran-ia/coreguapa-lm
> - GN Tweets LM: https://huggingface.co/guaran-ia/gntweets-lm :contentReference[oaicite:1]{index=1}


# Cálculo de la pérdida

Durante el procesamiento se calcula la pérdida media de cada ventana completa.

Todos los tokens reales de la ventana participan en el cálculo de la pérdida. Cuando dos ventanas se solapan, los tokens compartidos vuelven a contribuir en la ventana siguiente, de acuerdo con la receta publicada en los model cards de los modelos utilizados.

La pérdida media de cada ventana se pondera por la longitud completa de dicha ventana.

Conceptualmente, para cada ventana se acumula:

```text
pérdida_ponderada =
    pérdida_media_de_la_ventana
    ×
    longitud_de_la_ventana
```

La suma de estas pérdidas ponderadas se utiliza posteriormente para obtener la pérdida media global del documento.

---

# Cálculo de la perplexity

A medida que se procesan las ventanas, el pipeline acumula:

- la pérdida media de cada ventana multiplicada por la longitud completa de esa ventana;
- la suma de las longitudes de todas las ventanas procesadas.

Una vez finalizado el documento, la pérdida media global se obtiene mediante:

```text
average_negative_log_likelihood =
    total_negative_log_likelihood
    /
    total_window_tokens
```

La perplexity final se calcula aplicando la función exponencial:

```text
perplexity =
    exp(average_negative_log_likelihood)
```

Este procedimiento replica la receta de cálculo publicada en los model cards de `coreguapa-lm` y `gntweets-lm`.

---

# Uso de memoria

La implementación fue diseñada para mantener un consumo de memoria acotado incluso al procesar documentos muy largos.

Durante la ejecución permanecen en memoria:

- el documento JSON actualmente procesado;
- el fragmento de texto que está siendo tokenizado;
- el buffer correspondiente a la ventana activa;
- las ventanas pendientes del lote de inferencia;
- los tensores necesarios para ejecutar el modelo.

La implementación evita mantener simultáneamente en memoria una única secuencia completa de tokens correspondiente al documento completo.

Este enfoque permite procesar documentos cuya longitud excede ampliamente el tamaño máximo de contexto del modelo sin necesidad de transferir todo el documento tokenizado a la GPU.

---

# Ejecución del pipeline

El pipeline permite calcular cualquiera de las dos métricas de perplexity de forma independiente o calcular ambas durante una misma ejecución.

## Calcular únicamente CoreGuapa Perplexity

```bash
python -m src.pipeline.perplexity.run_perplexity_metrics --model coreguapa
```

---

## Calcular únicamente GN Tweets Perplexity

```bash
python -m src.pipeline.perplexity.run_perplexity_metrics --model tweets
```

---

## Calcular ambas métricas

```bash
python -m src.pipeline.perplexity.run_perplexity_metrics --model all
```

Cuando se utiliza la opción `all`, el pipeline calcula tanto `coreguapa_perplexity` como `tweets_perplexity` para todos los documentos del corpus.

---

# Reanudación de ejecuciones

Antes de calcular una métrica, el pipeline verifica si ésta ya existe en el documento correspondiente.

Si la métrica solicitada ya fue calculada, el documento no vuelve a procesarse para esa métrica.

Este comportamiento permite:

- reanudar ejecuciones interrumpidas;
- evitar trabajo duplicado;
- completar únicamente las métricas faltantes.

Por ejemplo, si una ejecución se interrumpe después de calcular `coreguapa_perplexity`, una ejecución posterior puede completar únicamente `tweets_perplexity` sin recalcular la primera.

---

# Validación

Una vez que ambas métricas han sido calculadas, el resultado puede verificarse mediante:

```bash
python -m src.pipeline.perplexity.validate_perplexity_metadata
```

El reporte de validación se almacena en:

```text
outputs/report/perplexity_metadata.log
```

Para cada corpus, el reporte incluye:

- número total de documentos;
- número de documentos que contienen `coreguapa_perplexity`;
- número de documentos que contienen `tweets_perplexity`;
- número de documentos que contienen ambas métricas.

Además, genera un resumen global indicando:

- número total de documentos procesados;
- cantidad de documentos con ambas métricas;
- cantidad de documentos con métricas faltantes;
- estado final de la validación.

> **Nota**
>
> El validador verifica que **todos los documentos del corpus contengan tanto `coreguapa_perplexity` como `tweets_perplexity`**.
>
> La validación debe ejecutarse únicamente cuando ambas métricas hayan sido calculadas, ya sea mediante:
>
> ```bash
> python -m src.pipeline.perplexity.run_perplexity_metrics --model all
> ```
>
> o ejecutando ambos comandos por separado:
>
> ```bash
> python -m src.pipeline.perplexity.run_perplexity_metrics --model coreguapa
> ```
>
> seguido de:
>
> ```bash
> python -m src.pipeline.perplexity.run_perplexity_metrics --model tweets
> ```
>
> Si solamente una de las dos métricas ha sido calculada, el resultado esperado de la validación será **FAIL**, ya que el estado final esperado del corpus requiere que todos los documentos contengan ambas métricas.

---

# Ejemplo de reporte

El archivo `perplexity_metadata.log` contiene un resumen por corpus y un resumen global de la validación.

Cada corpus informa:

- cantidad total de documentos;
- cantidad de documentos con `coreguapa_perplexity`;
- cantidad de documentos con `tweets_perplexity`;
- cantidad de documentos que contienen ambas métricas.

El resumen global incluye el número total de documentos procesados, la cantidad de documentos válidos y el estado final de la validación (`PASS` o `FAIL`).

---

# Limitaciones

El pipeline asume que:

- los documentos contienen un campo de texto válido;
- los modelos de Hugging Face pueden cargarse correctamente;
- existe memoria suficiente para cargar el modelo seleccionado.

Aunque la tokenización incremental reduce significativamente el consumo de memoria durante el procesamiento de documentos largos, el documento JSON correspondiente debe poder cargarse completamente en memoria antes de iniciar el cálculo.

La tokenización incremental debe conservar la misma secuencia de tokens que la tokenización completa para reproducir exactamente la receta del model card. La función de validación incluida en la implementación puede utilizarse con textos de prueba para comprobar esta equivalencia.

---

# Salida

Al finalizar la ejecución:

- los documentos JSONL contienen las métricas calculadas;
- las métricas previamente calculadas no son procesadas nuevamente;
- el validador genera un reporte con el estado de todos los corpus.

---

# Resumen

El pipeline proporciona un mecanismo para calcular métricas de perplexity sobre documentos de longitud arbitraria utilizando tokenización incremental y ventanas deslizantes.

Las principales características de la implementación son:

- cálculo de `coreguapa_perplexity` utilizando `guaran-ia/coreguapa-lm`;
- cálculo de `tweets_perplexity` utilizando `guaran-ia/gntweets-lm`;
- cálculo de la perplexity siguiendo la receta publicada en los model cards de los modelos;
- soporte para documentos más largos que la ventana de contexto del modelo;
- tokenización incremental para reducir el consumo de memoria;
- procesamiento de ventanas por lotes;
- reanudación automática de ejecuciones interrumpidas;
- validación de que todos los documentos contienen ambas métricas de perplexity.

---

# Referencias

## Modelos

- CoreGuapa LM: `guaran-ia/coreguapa-lm`
- GN Tweets LM: `guaran-ia/gntweets-lm`

## Código fuente

```text
src/pipeline/perplexity
```
