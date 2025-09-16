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
def load_market_data() -> tuple[pd.DataFrame, Dict[str, float]]:
    options_df = DataClient.fetch_filtered_options()
    underlying_prices = DataClient.get_underlying_prices()
    return options_df, underlying_prices


def process_market_data(
    options_df: pd.DataFrame,
    underlying_prices: Dict[str, float],
    risk_free_rate: float,
    dividend_yield: float,
) -> pd.DataFrame:
    config = ProcessorConfig(risk_free_rate=risk_free_rate, dividend_yield=dividend_yield)
    processor = OptionsProcessor(options_df, underlying_prices, config)
    return processor.enrich_with_greeks()


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
    options_df, underlying_prices = load_market_data()

if options_df.empty:
    st.error("❌ No se pudieron obtener datos del mercado")
    st.stop()

options_signature = dataframe_signature(options_df)
cache_key = (options_signature, risk_free_rate, dividend_yield)
if "processed_cache" not in st.session_state or st.session_state["processed_cache"]["key"] != cache_key:
    with st.spinner("⚙️ Procesando volatilidades implícitas y griegas..."):
        enriched_df = process_market_data(options_df, underlying_prices, risk_free_rate, dividend_yield)
    st.session_state["processed_cache"] = {"key": cache_key, "data": enriched_df.copy()}
else:
    enriched_df = st.session_state["processed_cache"]["data"].copy()

# Sidebar metrics
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Estado del Mercado")
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
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Contratos", len(enriched_df))
    col2.metric("IV Promedio", f"{enriched_df['iv'].mean():.1%}" if not enriched_df.empty else "N/A")
    col3.metric("Volumen Total", f"{enriched_df.get('v', pd.Series(dtype=float)).sum():,.0f}")
    col4.metric("Open Interest", f"{enriched_df.get('oi', pd.Series(dtype=float)).sum():,.0f}")

    has_underlying_data = not enriched_df.empty and "underlying" in enriched_df.columns

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
            summary_rows.append(
                {
                    "Subyacente": underlying,
                    "Prefijo": DataClient.TARGET_UNDERLYINGS[underlying],
                    "Precio": f"${underlying_prices.get(underlying, 0):.2f}",
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

    underlyings = sorted(enriched_df["underlying"].dropna().unique())
    if not underlyings:
        st.info("No hay datos para construir estrategias")
    else:
        col1, col2 = st.columns(2)
        with col1:
            selected_underlying = st.selectbox("Subyacente", underlyings)
        with col2:
            current_price = underlying_prices.get(selected_underlying, 100.0)
            st.metric(f"Precio actual {selected_underlying}", f"${current_price:.2f}")

        subset = enriched_df[enriched_df["underlying"] == selected_underlying]

        with st.expander("➕ Agregar nuevo leg"):
            leg_type = st.selectbox("Tipo", ["call", "put"], key="leg_type")
            leg_position = st.selectbox("Posición", ["long", "short"], key="leg_position")
            strikes = sorted(subset[subset["otype"] == leg_type]["K"].unique())
            leg_strike = st.selectbox("Strike", strikes if strikes else [current_price], key="leg_strike")
            leg_quantity = st.number_input("Cantidad", min_value=1, value=1, key="leg_quantity")
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

        if st.session_state["strategy_legs"]:
            legs = st.session_state["strategy_legs"]
            st.subheader("📋 Legs actuales")
            legs_df = pd.DataFrame([
                {
                    "Tipo": leg.option_type,
                    "Posición": leg.position,
                    "Strike": leg.strike,
                    "Prima": leg.premium,
                    "Cantidad": leg.quantity,
                    "IV": leg.iv,
                    "T": leg.T_original,
                }
                for leg in legs
            ])
            legs_df["Costo"] = [
                leg.premium * leg.quantity * (1 if leg.position == "long" else -1) for leg in legs
            ]
            st.dataframe(legs_df, use_container_width=True)
            total_cost = legs_df["Costo"].sum()
            st.metric("Costo total", f"${total_cost:.2f}")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🗑️ Limpiar estrategia"):
                    st.session_state["strategy_legs"] = []
                    st.rerun()
            with col_b:
                analysis_mode = st.radio(
                    "Modo de análisis",
                    ["Al vencimiento", "Valor teórico (días específicos)"],
                    horizontal=True,
                )

            price_min = st.number_input("Precio mínimo", value=current_price * 0.7)
            price_max = st.number_input("Precio máximo", value=current_price * 1.3)
            price_points = st.slider("Cantidad de puntos", 50, 200, 100)
            price_grid = np.linspace(price_min, price_max, price_points)
            days_ahead = 0
            if analysis_mode == "Valor teórico (días específicos)":
                days_ahead = st.slider("Días hacia adelante", 1, 90, 30)

            payoff = np.array([sum(leg.payoff(price, days_ahead) for leg in legs) for price in price_grid])
            render_payoff_chart(price_grid, payoff, current_price, days_ahead)

            max_profit = payoff.max()
            max_loss = payoff.min()
            current_pnl = np.interp(current_price, price_grid, payoff)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ganancia máxima", f"${max_profit:.2f}")
            col2.metric("Pérdida máxima", f"${max_loss:.2f}")
            col3.metric("P&L actual", f"${current_pnl:.2f}")
            roi = (current_pnl / abs(total_cost) * 100) if total_cost else 0
            col4.metric("ROI", f"{roi:.1f}%")

            st.subheader("🌪️ Sensibilidad a Volatilidad")
            if days_ahead > 0:
                vol_scenarios = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
                fig_vol = go.Figure()
                colors = ["blue", "green", "orange", "red", "purple", "brown"]
                for idx, vol in enumerate(vol_scenarios):
                    scenario_payoff = []
                    for price in price_grid:
                        pnl = 0.0
                        for leg in legs:
                            T_remaining = max(1 / 365, leg.T_original - days_ahead / 365)
                            theo_price = binomial_tree_american(price, leg.strike, T_remaining, leg.r, leg.q, vol, leg.option_type)
                            leg_pnl = theo_price - leg.premium
                            if leg.position == "short":
                                leg_pnl = -leg_pnl
                            pnl += leg_pnl * leg.quantity
                        scenario_payoff.append(pnl)
                    fig_vol.add_trace(
                        go.Scatter(
                            x=price_grid,
                            y=scenario_payoff,
                            mode="lines",
                            name=f"Vol {vol:.0%}",
                            line=dict(color=colors[idx % len(colors)]),
                        )
                    )
                fig_vol.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_vol.add_vline(x=current_price, line_dash="dot", line_color="black")
                fig_vol.update_layout(
                    title=f"Sensibilidad a volatilidad (en {days_ahead} días)",
                    xaxis_title="Precio del subyacente",
                    yaxis_title="P&L",
                    height=450,
                )
                st.plotly_chart(fig_vol, use_container_width=True)

            st.subheader("🎲 Simulación Monte Carlo")
            simulations = st.number_input("Simulaciones", value=5000, step=1000)
            days_mc = st.number_input("Días", value=30, step=5)
            underlying_vol = st.number_input("Volatilidad subyacente", value=0.25, step=0.01)
            if st.button("🚀 Ejecutar Monte Carlo"):
                with st.spinner("Ejecutando simulación..."):
                    pnls = monte_carlo_simulation(legs, current_price, int(simulations), int(days_mc), risk_free_rate, underlying_vol)
                expected_return = pnls.mean()
                volatility = pnls.std()
                var_95 = np.percentile(pnls, 5)
                var_99 = np.percentile(pnls, 1)
                cvar_95 = pnls[pnls <= var_95].mean()
                prob_profit = (pnls > 0).mean() * 100
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Retorno esperado", f"${expected_return:.2f}")
                col_b.metric("Volatilidad", f"${volatility:.2f}")
                col_c.metric("VaR 95%", f"${var_95:.2f}")
                col_d.metric("Prob. ganancia", f"{prob_profit:.1f}%")
                fig_hist = go.Figure()
                fig_hist.add_histogram(x=pnls, nbinsx=50)
                fig_hist.add_vline(x=expected_return, line_dash="dash", line_color="green", annotation_text="Media")
                fig_hist.add_vline(x=var_95, line_dash="dot", line_color="red", annotation_text="VaR 95%")
                fig_hist.update_layout(title="Distribución de P&L", xaxis_title="P&L", yaxis_title="Frecuencia", height=400)
                st.plotly_chart(fig_hist, use_container_width=True)

                metrics_df = pd.DataFrame(
                    {
                        "Métrica": [
                            "Retorno esperado",
                            "Volatilidad",
                            "VaR 95%",
                            "VaR 99%",
                            "CVaR 95%",
                            "Prob. ganancia",
                            "Máx. ganancia",
                            "Máx. pérdida",
                        ],
                        "Valor": [
                            f"${expected_return:.2f}",
                            f"${volatility:.2f}",
                            f"${var_95:.2f}",
                            f"${var_99:.2f}",
                            f"${cvar_95:.2f}",
                            f"{prob_profit:.1f}%",
                            f"${pnls.max():.2f}",
                            f"${pnls.min():.2f}",
                        ],
                    }
                )
                st.dataframe(metrics_df, use_container_width=True)

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
