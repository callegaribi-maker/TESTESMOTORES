"""
sit_to_stand.py — Módulo Sentar e Levantar (Sit-to-Stand) (Momentum Web)

Protocolo: celular no tronco (peito ou lombar). Paciente realiza repetições
de sentar-levantar (5 repetições cronometradas, ou 30s de repetições
máximas). Arquivo CSV/TXT: Tempo, X, Y, Z (acelerômetro).
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks

import common


def render():
    st.subheader("🪑 Sentar e Levantar (Sit-to-Stand)")
    st.caption(
        "Celular no tronco (peito ou região lombar) durante as repetições. "
        "Envie o arquivo CSV/TXT do sensor inercial (Tempo, X, Y, Z)."
    )

    modo = st.radio("Protocolo", ["5 repetições cronometradas", "Repetições máximas em 30s"], horizontal=True)

    data = common.get_uploaded_series("Selecione o arquivo CSV/TXT", "sit_to_stand")
    if data is None:
        return

    t_sec, x, y, z = data
    fs = common.sampling_rate(t_sec)

    eixo = st.selectbox("Eixo vertical (dominante no movimento)", ["X", "Y", "Z", "SVM (módulo)"], index=3)
    sig_map = {"X": x, "Y": y, "Z": z}
    sig = np.sqrt(x ** 2 + y ** 2 + z ** 2) if eixo == "SVM (módulo)" else sig_map[eixo]

    sig_dt = common.detrend(sig)
    sig_f = common.lowpass_filter(sig_dt, fs, min(5.0, fs / 2 - 0.5)) if fs > 2 else sig_dt

    col_a, col_b = st.columns(2)
    with col_a:
        prominence = st.slider("Proeminência mínima do pico", 0.05, float(max(2.0, np.std(sig_f) * 3)),
                                float(np.std(sig_f) * 0.6), 0.05)
    with col_b:
        min_interval = st.slider("Intervalo mínimo entre repetições (s)", 0.3, 3.0, 0.8, 0.1)

    distance = max(int(fs * min_interval), 1) if fs > 0 else 1
    peaks, _ = find_peaks(sig_f, distance=distance, prominence=prominence)

    markers = [{"x": t_sec[p], "label": f"{i+1}", "color": "#c00000"} for i, p in enumerate(peaks)]
    st.plotly_chart(
        common.plot_timeseries(t_sec, {f"Sinal ({eixo})": sig_f}, yaxis_title="Aceleração", markers=markers),
        use_container_width=True,
    )

    n_reps = len(peaks)
    st.markdown("#### Resultados")

    if n_reps < 2:
        st.warning("Menos de 2 repetições detectadas. Ajuste a proeminência/intervalo acima.")
        return

    rep_times = t_sec[peaks]
    total_time = rep_times[-1] - rep_times[0]
    cycle_durations = np.diff(rep_times)
    cadence = n_reps / (t_sec[-1] - t_sec[0]) * 60 if (t_sec[-1] - t_sec[0]) > 0 else np.nan

    resultados = {
        "Repetições detectadas": n_reps,
        "Tempo total (1ª à última repetição) (s)": total_time,
        "Duração média por ciclo (s)": float(np.mean(cycle_durations)),
        "Desvio-padrão do ciclo (s)": float(np.std(cycle_durations)),
        "Cadência (repetições/min)": cadence,
    }
    df = pd.DataFrame(resultados.items(), columns=["Métrica", "Valor"])
    df["Valor"] = df["Valor"].map(lambda v: f"{v:.2f}" if isinstance(v, (int, float, np.floating)) else v)
    st.dataframe(df, use_container_width=True, hide_index=True)

    if modo == "5 repetições cronometradas":
        if n_reps != 5:
            st.warning(f"Foram detectadas {n_reps} repetições, e não 5. Confira o sinal/ajuste os parâmetros.")
        else:
            st.info(f"**Tempo do teste (5 repetições): {total_time:.2f} s**")
    else:
        st.info(f"**{n_reps} repetições completas em ~{t_sec[-1] - t_sec[0]:.1f} s** (cadência {cadence:.1f} rep/min)")

    st.caption("Ajuste manualmente a proeminência/intervalo se a contagem automática não bater com o número real de repetições.")
