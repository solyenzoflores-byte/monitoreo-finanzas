# Monitor de Opciones Financieras

Este proyecto proporciona un monitor en tiempo real para opciones
argentinas junto con un proceso separado para almacenar datos
históricos en SQLite.

## Componentes principales

- **`app.py`**: dashboard interactivo construido con Streamlit.
- **`historical_ingestor.py`**: script de línea de comandos para
  descargar la información y guardarla en la base de datos.
- **Módulos en `core/`**: utilidades compartidas entre ambos flujos
  (cliente HTTP, procesador de opciones y administrador de base de
  datos).

## Requisitos

- Python 3.10+
- Dependencias listadas en `requirements.txt` (puedes crear un entorno
  virtual y ejecutar `pip install -r requirements.txt`).

## Uso

### Dashboard en tiempo real

```bash
streamlit run app.py
```

El panel permite refrescar datos manualmente o configurar un
auto-refresco. En la pestaña de “Estrategias” se pueden construir legs y
analizar el payoff, sensibilidad a la volatilidad y ejecutar simulaciones
Monte Carlo. La pestaña “Base de Datos Histórica” consulta registros
guardados en SQLite.

#### Fuentes de datos y fallbacks

Por defecto las cotizaciones se descargan desde `data912.com`. Si el
entorno impide acceder a ese dominio (por ejemplo, cuando solo se permite
salir por GitHub), la aplicación cargará automáticamente un snapshot
incluido en `core/sample_data`. Se mostrará un mensaje en la interfaz
avisando que se está utilizando dicho respaldo y el motivo del error.

Puedes definir endpoints alternativos mediante variables de entorno; por
ejemplo:

```bash
export MONITOREO_OPCIONES_SOURCES="https://tu.dominio/live/arg_options.json"
export MONITOREO_ACCIONES_SOURCES="https://tu.dominio/live/arg_stocks.json"
```

Los valores pueden contener varias URLs separadas por comas y se intentan
en orden hasta conseguir una respuesta válida.

### Ingesta de datos históricos

```bash
python historical_ingestor.py --interval 120
```

Al ejecutar el script se descargan los datos de mercado, se calculan las
métricas necesarias y se guardan en la base `options_data.db`. Con el
parámetro `--interval` se ejecuta en bucle, esperando la cantidad de
segundos indicada entre cada ciclo. Si se omite dicho parámetro, realiza
una única descarga.
