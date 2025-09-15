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

### Ingesta de datos históricos

```bash
python historical_ingestor.py --interval 120
```

Al ejecutar el script se descargan los datos de mercado, se calculan las
métricas necesarias y se guardan en la base `options_data.db`. Con el
parámetro `--interval` se ejecuta en bucle, esperando la cantidad de
segundos indicada entre cada ciclo. Si se omite dicho parámetro, realiza
una única descarga.
