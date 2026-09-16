"""
static_balance.py — Módulo de Equilíbrio Estático (Momentum Web)

Protocolo esperado: celular fixado próximo ao centro de massa (cintura/tronco
baixo), paciente em pé e parado por ~20-30s. Arquivo CSV/TXT: Tempo, X, Y, Z
(acelerômetro, em m/s² ou g).
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import common


def _sway_metrics(x: np.ndarray, y: np.ndarray, t_sec: np.ndarray) -> dict:
    duration = t_sec[-1] - t_sec[0] if len(t_sec) > 1 else np.nan

    # Path length (comprimento do deslocamento no plano X-Y) — proxy de sway
    path = np.sum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    mean_velocity = path / duration if duration else np.nan

    rms_x = float(np.sqrt(np.mean(x ** 2)))
    rms_y = float(np.sqrt(np.mean(y ** 2)))
    range_x = float(np.ptp(x))
    range_y = float(np.ptp(y))

    # Área da elipse de confiança 95% (PCA no plano X-Y)
    cov = np.cov(x, y)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    chi2_95 = 5.991  # qui-quadrado, 2 graus de liberdade, 95%
    ellipse_area = float(np.pi * chi2_95 * np.sqrt(eigvals[0] * eigvals[1]))

    return {
        "Duração (s)": duration,
        "RMS X (m/s²)": rms_x,
        "RMS Y (m/s²)": rms_y,
        "Amplitude X (m/s²)": range_x,
        "Amplitude Y (m/s²)": range_y,
        "Comprimento do sway (u.a.)": float(path),
        "Velocidade média de sway (u.a./s)": float(mean_velocity),
        "Área da elipse 95% (u.a.²)": ellipse_area,
    }


def render():
    st.subheader("⚖️ Equilíbrio Estático")
    st.caption(
        "Celular fixado próximo ao centro de massa (cintura), paciente em pé, parado. "
        "Envie o arquivo CSV/TXT exportado do sensor inercial (Tempo, X, Y, Z)."
    )

    condicao = st.selectbox("Condição do teste", ["Olhos abertos", "Olhos fechados", "Outra"])

    data = common.get_uploaded_series("Selecione o arquivo CSV/TXT", "static_balance")
    if data is None:
        return

    t_sec, x, y, z = data
    fs = common.sampling_rate(t_sec)

    col_a, col_b = st.columns(2)
    with col_a:
        apply_detrend = st.checkbox("Aplicar detrend (remover offset/gravidade)", True)
    with col_b:
        apply_filter = st.checkbox("Aplicar filtro passa-baixa", True)

    xf, yf, zf = x.copy(), y.copy(), z.copy()
    if apply_detrend:
        xf, yf, zf = common.detrend(xf), common.detrend(yf), common.detrend(zf)

    if apply_filter and fs > 0:
        cutoff = st.slider("Frequência de corte (Hz)", 0.5, min(20.0, fs / 2 - 0.5), 10.0, 0.5)
        xf = common.lowpass_filter(xf, fs, cutoff)
        yf = common.lowpass_filter(yf, fs, cutoff)
        zf = common.lowpass_filter(zf, fs, cutoff)

    st.plotly_chart(
        common.plot_timeseries(t_sec, {"X": xf, "Y": yf, "Z": zf}, yaxis_title="Aceleração"),
        use_container_width=True,
    )

    st.markdown("#### Estatocinesiograma (X vs Y)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xf, y=yf, mode="lines+markers",
                              marker=dict(size=3), line=dict(width=1)))
    fig.update_layout(
        xaxis=dict(title="X", showline=True, linecolor="black", mirror=True, scaleanchor="y"),
        yaxis=dict(title="Y", showline=True, linecolor="black", mirror=True),
    )
    st.plotly_chart(fig, use_container_width=True)

    metrics = _sway_metrics(xf, yf, t_sec)
    st.markdown(f"#### Resultados — {condicao}")
    df_metrics = pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"])
    df_metrics["Valor"] = df_metrics["Valor"].map(lambda v: f"{v:.4f}")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.caption(
        "Métricas calculadas a partir da aceleração (não do centro de pressão). "
        "Use-as de forma comparativa (ex.: olhos abertos vs. fechados, pré vs. pós)."
    )
