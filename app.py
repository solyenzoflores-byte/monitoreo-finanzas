"""Streamlit dashboard for monitoring Argentine options."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from core.data_client import DataClient
from core.database import DatabaseManager
from core.processing import OptionsProcessor, ProcessorConfig
from core.models import binomial_tree_american


st.set_page_config(
    page_title="Monitor de Opciones Pro - ARG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@dataclass
class StrategyLeg:
    option_type: str
    position: str
    strike: float
    premium: float
    quantity: int
    iv: float
    T_original: float
    r: float
    q: float

    def payoff(self, price: float, days_ahead: int = 0) -> float:
        if days_ahead <= 0:
            intrinsic = max(0.0, price - self.strike) if self.option_type == "call" else max(0.0, self.strike - price)
            value = intrinsic
        else:
            T_remaining = max(1 / 365, self.T_original - days_ahead / 365)
            value = binomial_tree_american(price, self.strike, T_remaining, self.r, self.q, self.iv, self.option_type)
        pnl = value - self.premium
        if self.position == "short":
            pnl = -pnl
        return pnl * self.quantity


def dataframe_signature(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha1(hashed).hexdigest()


@st.cache_resource
def get_db() -> DatabaseManager:
    return DatabaseManager()


@st.cache_data(ttl=10)
def load_market_data() -> tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    options_df = DataClient.fetch_filtered_options()
    underlying_prices = DataClient.get_underlying_prices()
    fx_rates = DataClient.get_exchange_rates()
    return options_df, underlying_prices, fx_rates


def process_market_data(
    options_df: pd.DataFrame,
    underlying_prices: Dict[str, float],
    risk_free_rate: float,
    dividend_yield: float,
) -> pd.DataFrame:
    config = ProcessorConfig(risk_free_rate=risk_free_rate, dividend_yield=dividend_yield)
    processor = OptionsProcessor(options_df, underlying_prices, config)
    return processor.enrich_with_greeks()


def pick_preferred_fx_rate(fx_rates: Dict[str, float]) -> float | None:
    if not fx_rates:
        return None
    priority_keys = ("mep", "usd", "dolar", "dólar", "blue", "ccl", "oficial", "promedio")
    for fragment in priority_keys:
        for key, value in fx_rates.items():
            if fragment in key.lower():
                return value
    return next(iter(fx_rates.values()), None)


def show_greek_tooltip() -> None:
    with st.expander("ℹ️ Información sobre las Griegas"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                **Delta (Δ)**: Sensibilidad del precio de la opción al cambio en el precio del subyacente.
                - Call: 0 a 1
                - Put: -1 a 0
                - Delta hedging requiere Δ × 100 acciones por contrato

                **Gamma (Γ)**: Tasa de cambio del Delta. Máxima en opciones ATM.
                - Mide la aceleración del precio de la opción
                - Importante para gestión de riesgo en posiciones delta-neutral

                **Vega (ν)**: Sensibilidad a cambios en volatilidad implícita.
                - Expresada por 1% de cambio en IV
                - Mayor en opciones ATM y con más tiempo al vencimiento
                """
            )
        with col2:
            st.markdown(
                """
                **Theta (Θ)**: Decaimiento temporal (time decay).
                - Generalmente negativa para compras de opciones
                - Se acelera cerca del vencimiento
                - Mayor impacto en opciones ATM

                **Rho (ρ)**: Sensibilidad a cambios en tasas de interés.
                - Más relevante en opciones con vencimientos largos
                - Calls: positivo, Puts: negativo

                **IV**: Volatilidad "implícita" en el precio de mercado.
                - Refleja expectativas futuras del mercado
                - Diferente de la volatilidad histórica
                """
            )


def render_payoff_chart(prices: np.ndarray, payoff: np.ndarray, current_price: float, days_ahead: int) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices, y=payoff, mode="lines", name="Payoff", line=dict(color="green", width=3)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=current_price, line_dash="dot", line_color="red", annotation_text=f"S=${current_price:.2f}")
    sign_changes = np.where(np.diff(np.sign(payoff)) != 0)[0]
    for idx in sign_changes:
        be_price = prices[idx]
        fig.add_vline(x=be_price, line_dash="dashdot", line_color="orange", annotation_text=f"BE ${be_price:.2f}")
    title_suffix = f" (en {days_ahead} días)" if days_ahead > 0 else " (al vencimiento)"
    fig.update_layout(
        title=f"Payoff de la Estrategia{title_suffix}",
        xaxis_title="Precio del Subyacente",
        yaxis_title="P&L",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def monte_carlo_simulation(
    legs: List[StrategyLeg],
    current_price: float,
    simulations: int,
    days: int,
    risk_free: float,
    underlying_vol: float,
) -> np.ndarray:
    dt = days / 365.0
    np.random.seed(42)
    drift = (risk_free - 0.5 * underlying_vol**2) * dt
    shocks = np.random.normal(drift, underlying_vol * np.sqrt(dt), simulations)
    prices = current_price * np.exp(shocks)
    pnls = np.array([sum(leg.payoff(price, days) for leg in legs) for price in prices])
    return pnls


# Sidebar configuration
st.sidebar.title("⚙️ Configuración Global")
if st.sidebar.button("🔃 Actualizar datos"):
    DataClient.clear_cache()
    load_market_data.clear()
    st.rerun()

auto_refresh = st.sidebar.checkbox("🔄 Auto-actualizar", value=False)
if auto_refresh:
    refresh_interval = st.sidebar.slider("Intervalo (segundos)", 10, 60, 30)
else:
    refresh_interval = 0

with st.sidebar.expander("📊 Parámetros del Modelo"):
    risk_free_rate = st.number_input("Tasa libre de riesgo (%)", value=5.0, step=0.1) / 100
    dividend_yield = st.number_input("Rendimiento dividendos (%)", value=0.0, step=0.1) / 100

market_open = st.sidebar.time_input("🕘 Apertura mercado", value=datetime.strptime("11:00", "%H:%M").time())
market_close = st.sidebar.time_input("🕕 Cierre mercado", value=datetime.strptime("17:00", "%H:%M").time())

with st.spinner("📡 Obteniendo datos del mercado..."):
    options_df, underlying_prices, fx_rates = load_market_data()

if options_df.empty:
    st.error("❌ No se pudieron obtener datos del mercado")
    st.stop()

preferred_fx_rate = pick_preferred_fx_rate(fx_rates)
default_exchange_value = (
    float(preferred_fx_rate)
    if preferred_fx_rate is not None and preferred_fx_rate > 0
    else float(st.session_state.get("exchange_rate", 1000.0))
)
exchange_rate = st.sidebar.number_input(
    "💱 Tipo de cambio (ARS/USD)",
    min_value=0.0,
    value=float(st.session_state.get("exchange_rate", default_exchange_value)),
    step=1.0,
    format="%.2f",
)
st.session_state["exchange_rate"] = float(exchange_rate)
if preferred_fx_rate is not None:
    st.sidebar.caption(f"Referencia de mercado (MEP/CCL): ${preferred_fx_rate:,.2f}")

options_signature = dataframe_signature(options_df)
cache_key = (options_signature, risk_free_rate, dividend_yield)
if "processed_cache" not in st.session_state or st.session_state["processed_cache"]["key"] != cache_key:
    with st.spinner("⚙️ Procesando volatilidades implícitas y griegas..."):
        enriched_df = process_market_data(options_df, underlying_prices, risk_free_rate, dividend_yield)
    st.session_state["processed_cache"] = {"key": cache_key, "data": enriched_df.copy()}
else:
    enriched_df = st.session_state["processed_cache"]["data"].copy()

target_underlyings = set(DataClient.TARGET_UNDERLYINGS.keys())
has_underlying_data = (
    not enriched_df.empty
    and "underlying" in enriched_df.columns
    and enriched_df["underlying"].isin(target_underlyings).any()
)

# Sidebar metrics
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Estado del Mercado")
st.sidebar.metric("Tipo de cambio utilizado", f"${exchange_rate:,.2f}")
if preferred_fx_rate is not None:
    st.sidebar.metric("Dólar referencia", f"${preferred_fx_rate:,.2f}")
for underlying, price in underlying_prices.items():
    st.sidebar.metric(underlying, f"${price:.2f}")
st.sidebar.metric("Contratos disponibles", len(enriched_df))

if auto_refresh and refresh_interval:
    time.sleep(refresh_interval)
    st.rerun()


tab_dashboard, tab_analysis, tab_strategy, tab_database = st.tabs(
    [
        "📊 Dashboard Principal",
        "🔍 Análisis IV & Griegas",
        "🎯 Estrategias & Riesgo",
        "📚 Base de Datos Histórica",
    ]
)

with tab_dashboard:
    st.title("📊 Monitor de Opciones Americanas - Argentina")
    st.markdown("Análisis para **ALUA (ALU)**, **GGAL (GFG)** y **COME (COM)**")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Contratos", len(enriched_df))
    col2.metric("IV Promedio", f"{enriched_df['iv'].mean():.1%}" if not enriched_df.empty else "N/A")
    col3.metric("Volumen Total", f"{enriched_df.get('v', pd.Series(dtype=float)).sum():,.0f}")
    col4.metric("Open Interest", f"{enriched_df.get('oi', pd.Series(dtype=float)).sum():,.0f}")
    exchange_metric_value = f"${exchange_rate:,.2f}" if exchange_rate > 0 else "N/A"
    col5.metric("Tipo de cambio (ARS/USD)", exchange_metric_value)


    st.subheader("📋 Resumen por Subyacente")
    if not has_underlying_data:
        st.info("No hay datos disponibles para generar el resumen por subyacente.")
    else:
        summary_rows = []
        for underlying in DataClient.TARGET_UNDERLYINGS.keys():
            subset = enriched_df[enriched_df["underlying"] == underlying]
            if subset.empty:
                continue
            calls = subset[subset["otype"] == "call"]
            puts = subset[subset["otype"] == "put"]
            price_ars = float(underlying_prices.get(underlying, 0))
            price_usd = price_ars / exchange_rate if exchange_rate and exchange_rate > 0 else np.nan
            summary_rows.append(
                {
                    "Subyacente": underlying,
                    "Prefijo": DataClient.TARGET_UNDERLYINGS[underlying],
                    "Precio": f"${price_ars:.2f}",
                    "Precio USD": f"U$D {price_usd:.2f}" if not np.isnan(price_usd) else "N/A",
                    "Calls": len(calls),
                    "Puts": len(puts),
                    "IV Calls": f"{calls['iv'].mean():.1%}" if not calls.empty else "N/A",
                    "IV Puts": f"{puts['iv'].mean():.1%}" if not puts.empty else "N/A",
                    "Volumen": int(subset.get("v", pd.Series(dtype=float)).sum()),
                }
            )
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
        else:
            st.info("No se encontraron datos para los subyacentes objetivo.")

    st.subheader("💱 Tipo de cambio y liquidez")
    fx_col1, fx_col2 = st.columns(2)
    fx_col1.metric("Tipo de cambio utilizado", exchange_metric_value)
    if preferred_fx_rate is not None:
        fx_col2.metric("Referencia de mercado", f"${preferred_fx_rate:,.2f}")
    elif fx_rates:
        first_key, first_value = next(iter(fx_rates.items()))
        fx_col2.metric(first_key, f"${first_value:,.2f}")
    else:
        with fx_col2:
            st.info("Sin datos de referencia")

    if fx_rates:
        with st.expander("Ver detalle de tipos de cambio disponibles"):
            fx_display = (
                pd.DataFrame(
                    [
                        {"Fuente": key, "ARS por USD": float(value)}
                        for key, value in fx_rates.items()
                    ]
                )
                .sort_values("Fuente")
                .reset_index(drop=True)
            )
            st.dataframe(
                fx_display.style.format({"ARS por USD": "{:,.2f}"}),
                use_container_width=True,
            )

    if "liquidity_ars" not in st.session_state:
        st.session_state["liquidity_ars"] = 0.0
    if "liquidity_usd" not in st.session_state:
        st.session_state["liquidity_usd"] = 0.0

    with st.expander("Configurar liquidez en ARS y USD", expanded=True):
        col_liq_ars, col_liq_usd = st.columns(2)
        ars_liquidity = col_liq_ars.number_input(
            "Liquidez en pesos (ARS)",
            min_value=0.0,
            step=1000.0,
            format="%.2f",
            key="liquidity_ars",
        )
        usd_liquidity = col_liq_usd.number_input(
            "Liquidez en dólares (USD)",
            min_value=0.0,
            step=10.0,
            format="%.2f",
            key="liquidity_usd",
        )

        if exchange_rate and exchange_rate > 0:
            usd_in_ars = usd_liquidity * exchange_rate
            total_ars = ars_liquidity + usd_in_ars
            total_usd = total_ars / exchange_rate if exchange_rate else 0.0
            composition_data = pd.DataFrame(
                [
                    {
                        "Moneda": "ARS",
                        "Monto nominal": ars_liquidity,
                        "Equivalente ARS": ars_liquidity,
                        "Equivalente USD": ars_liquidity / exchange_rate,
                    },
                    {
                        "Moneda": "USD",
                        "Monto nominal": usd_liquidity,
                        "Equivalente ARS": usd_in_ars,
                        "Equivalente USD": usd_liquidity,
                    },
                ]
            )
            if total_ars > 0:
                composition_data["Participación (%)"] = (
                    composition_data["Equivalente ARS"] / total_ars * 100.0
                )
            else:
                composition_data["Participación (%)"] = 0.0

            st.dataframe(
                composition_data.style.format(
                    {
                        "Monto nominal": "{:,.2f}",
                        "Equivalente ARS": "{:,.2f}",
                        "Equivalente USD": "{:,.2f}",
                        "Participación (%)": "{:,.2f}",
                    }
                ),
                use_container_width=True,
            )

            summary_cols = st.columns(2)
            summary_cols[0].metric("Total liquidez (ARS)", f"${total_ars:,.2f}")
            summary_cols[1].metric("Total liquidez (USD)", f"U$D {total_usd:,.2f}")

            usd_share = (
                float(
                    composition_data.loc[
                        composition_data["Moneda"] == "USD", "Participación (%)"
                    ].iloc[0]
                )
                if total_ars > 0
                else 0.0
            )
            st.caption(
                f"Los dólares equivalen a ${usd_in_ars:,.2f} y representan {usd_share:.2f}% de la liquidez total."
            )
        else:
            st.info("Definí un tipo de cambio mayor que cero para calcular la composición.")

    current_time = datetime.now().time()
    if market_open <= current_time <= market_close:
        with st.spinner("💾 Guardando datos históricos"):
            db = get_db()
            db.save_underlying_prices(underlying_prices)
            if has_underlying_data:
                for underlying in DataClient.TARGET_UNDERLYINGS.keys():
                    subset = enriched_df[enriched_df["underlying"] == underlying]
                    if subset.empty:
                        continue
                    db.save_options_data(subset, underlying)
        if has_underlying_data:
            st.success("✅ Datos almacenados en la base de datos")
        else:
            st.info("No hay datos de opciones para almacenar en la base de datos en este momento.")

with tab_analysis:
    st.title("🔍 Análisis de Volatilidad Implícita y Griegas")
    show_greek_tooltip()
    if enriched_df.empty:
        st.info("No hay datos disponibles para análisis")
    else:
        underlyings = ["Todos"] + sorted(enriched_df["underlying"].dropna().unique())
        selected_underlying = st.selectbox("Seleccionar Subyacente", underlyings)
        if selected_underlying != "Todos":
            analysis_df = enriched_df[enriched_df["underlying"] == selected_underlying]
            current_price = underlying_prices.get(selected_underlying, 100.0)
            if exchange_rate and exchange_rate > 0:
                st.caption(
                    f"Precio spot {selected_underlying}: ${current_price:.2f} | U$D {current_price / exchange_rate:.2f} (TC {exchange_rate:,.2f})"
                )
            else:
                st.caption(f"Precio spot {selected_underlying}: ${current_price:.2f}")
        else:
            analysis_df = enriched_df
            current_price = np.nan

        if analysis_df.empty:
            st.warning("Sin datos para el subyacente seleccionado")
        else:
            st.subheader("😊 Volatility Smile")
            fig_smile = go.Figure()
            for option_type, color in (("call", "blue"), ("put", "red")):
                type_df = analysis_df[analysis_df["otype"] == option_type]
                if type_df.empty:
                    continue
                fig_smile.add_trace(
                    go.Scatter(
                        x=type_df["moneyness"],
                        y=type_df["iv"],
                        mode="markers",
                        name=option_type.upper(),
                        marker=dict(color=color, size=8),
                    )
                )
            fig_smile.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="ATM")
            fig_smile.update_layout(
                xaxis_title="Moneyness (S/K)",
                yaxis_title="Volatilidad Implícita",
                height=400,
            )
            st.plotly_chart(fig_smile, use_container_width=True)

            greek = st.selectbox("Seleccionar Griega", ["delta", "gamma", "vega", "theta", "rho"])
            fig_greek = go.Figure()
            for option_type, color in (("call", "blue"), ("put", "red")):
                type_df = analysis_df[analysis_df["otype"] == option_type]
                if type_df.empty or greek not in type_df.columns:
                    continue
                fig_greek.add_trace(
                    go.Scatter(
                        x=type_df["K"],
                        y=type_df[greek],
                        mode="markers+lines",
                        name=f"{option_type.upper()} {greek}",
                        marker=dict(color=color),
                    )
                )
            if not np.isnan(current_price):
                fig_greek.add_vline(x=current_price, line_dash="dash", line_color="gray", annotation_text=f"S=${current_price:.2f}")
            fig_greek.update_layout(
                xaxis_title="Strike",
                yaxis_title=greek.capitalize(),
                height=400,
            )
            st.plotly_chart(fig_greek, use_container_width=True)

            display_cols = [
                "symbol",
                "underlying",
                "otype",
                "K",
                "mkt_price",
                "iv",
                "delta",
                "gamma",
                "vega",
                "theta",
                "time_value",
                "moneyness",
            ]
            show_df = analysis_df[display_cols].copy()
            if not show_df.empty:
                show_df["iv"] = show_df["iv"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                for col in ["delta", "gamma", "vega", "theta"]:
                    show_df[col] = show_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
                st.dataframe(show_df, use_container_width=True, height=400)

            if "expiration" in analysis_df.columns:
                st.subheader("📈 Estructura Temporal de IV")
                term_structure = analysis_df.dropna(subset=["expiration"]).groupby("expiration")["iv"].mean().reset_index()
                term_structure["days_to_exp"] = term_structure["expiration"].apply(lambda x: (x - pd.Timestamp.now()).days)
                term_structure = term_structure.sort_values("days_to_exp")
                if term_structure.empty:
                    st.info("Sin vencimientos disponibles para graficar")
                else:
                    fig_term = go.Figure()
                    fig_term.add_trace(
                        go.Scatter(
                            x=term_structure["days_to_exp"],
                            y=term_structure["iv"],
                            mode="markers+lines",
                            line=dict(color="purple"),
                        )
                    )
                    fig_term.update_layout(
                        xaxis_title="Días al vencimiento",
                        yaxis_title="IV promedio",
                        height=350,
                    )
                    st.plotly_chart(fig_term, use_container_width=True)

with tab_strategy:
    st.title("🎯 Simulador de Estrategias y Gestión de Riesgo")
    if "strategy_legs" not in st.session_state:
        st.session_state["strategy_legs"] = []
    # TODO: reconstruir panel de estrategias; bloque anterior deshabilitado temporalmente.
    if False:  # pragma: no cover - mantiene el código previo sin ejecutarlo
        option_row = subset[(subset["otype"] == leg_type) & (subset["K"] == leg_strike)]
        if option_row.empty:
            st.warning("No se encontró información para ese strike; usando valores por defecto")
            premium = 1.0
            iv = 0.25
            T_value = 30 / 365
        else:
            premium = float(option_row["mkt_price"].iloc[0])
            iv = float(option_row["iv"].iloc[0])
            T_value = float(option_row["T"].iloc[0]) if "T" in option_row.columns else 30 / 365
            st.info(f"Precio actual: ${premium:.2f} | IV: {iv:.1%}")
        if st.button("Agregar leg"):
            leg = StrategyLeg(
                option_type=leg_type,
                position=leg_position,
                strike=float(leg_strike),
                premium=premium,
                quantity=int(leg_quantity),
                iv=iv,
                T_original=T_value,
                r=risk_free_rate,
                q=dividend_yield,
            )
            st.session_state["strategy_legs"].append(leg)
            st.success("Leg agregado correctamente")
            st.rerun()


with tab_database:
    st.title("📚 Base de Datos Histórica")
    db = get_db()
    analysis_type = st.selectbox(
        "Tipo de análisis",
        [
            "Volatilidad Implícita",
            "Volumen y Open Interest",
            "Precios Históricos",
        ],
    )
    selected_underlying = st.selectbox("Subyacente", list(DataClient.TARGET_UNDERLYINGS.keys()))
    days = st.selectbox("Período", [7, 15, 30, 60, 90], index=2)
    if st.button("🔍 Consultar"):
        with st.spinner("Consultando base de datos..."):
            if analysis_type == "Volatilidad Implícita":
                data = db.get_historical_iv(selected_underlying, days)
                if data.empty:
                    st.warning("No hay datos para el período seleccionado")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["iv"], mode="lines"))
                    fig.update_layout(title=f"IV promedio {selected_underlying}", xaxis_title="Fecha", yaxis_title="IV")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(data, use_container_width=True)
            elif analysis_type == "Volumen y Open Interest":
                data = db.get_volume_open_interest(selected_underlying, days)
                if data.empty:
                    st.warning("No hay datos para el período seleccionado")
                else:
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    fig.add_trace(go.Bar(x=data["timestamp"], y=data["volume"], name="Volumen"), secondary_y=False)
                    fig.add_trace(
                        go.Scatter(x=data["timestamp"], y=data["open_interest"], name="Open Interest", mode="lines"),
                        secondary_y=True,
                    )
                    fig.update_layout(title=f"Volumen y Open Interest {selected_underlying}")
                    fig.update_xaxes(title_text="Fecha")
                    fig.update_yaxes(title_text="Volumen", secondary_y=False)
                    fig.update_yaxes(title_text="Open Interest", secondary_y=True)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(data, use_container_width=True)
            else:
                data = db.get_underlying_history(selected_underlying, days)
                if data.empty:
                    st.warning("No hay datos para el período seleccionado")
                else:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data["timestamp"], y=data["price"], mode="lines"))
                    fig.update_layout(title=f"Precio histórico {selected_underlying}", xaxis_title="Fecha", yaxis_title="Precio")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(data, use_container_width=True)
