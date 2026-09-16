"""
resting_tremor.py — Módulo de Tremor de Repouso (Momentum Web)

Protocolo: celular fixado ao dorso da mão/antebraço, membro totalmente
relaxado e apoiado (repouso), sem contração voluntária, por ~15-30s.
Arquivo CSV/TXT: Tempo, X, Y, Z (acelerômetro).
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import welch

import common

TREMOR_BAND = (2.0, 12.0)  # Hz — faixa geral de tremores patológicos
PARKINSON_BAND = (4.0, 6.0)


def render():
    st.subheader("✋ Tremor de Repouso")
    st.caption(
        "Celular fixado ao dorso da mão/antebraço, membro relaxado e apoiado, sem contração voluntária. "
        "Envie o arquivo CSV/TXT do sensor inercial (Tempo, X, Y, Z)."
    )

    data = common.get_uploaded_series("Selecione o arquivo CSV/TXT", "resting_tremor")
    if data is None:
        return

    t_sec, x, y, z = data
    fs = common.sampling_rate(t_sec)
    if fs <= 0:
        st.error("Não foi possível estimar a taxa de amostragem do arquivo.")
        return
    st.caption(f"Taxa de amostragem estimada: **{fs:.1f} Hz**")

    eixo = st.selectbox("Sinal analisado", ["SVM (módulo, recomendado)", "X", "Y", "Z"], index=0)
    sig = np.sqrt(x ** 2 + y ** 2 + z ** 2) if eixo.startswith("SVM") else {"X": x, "Y": y, "Z": z}[eixo]
    sig = common.detrend(sig)

    low, high = st.slider("Faixa de filtragem (Hz)", 0.5, min(20.0, fs / 2 - 0.5), TREMOR_BAND, 0.5)
    sig_f = common.bandpass_filter(sig, fs, low, high)

    st.plotly_chart(
        common.plot_timeseries(t_sec, {"Sinal bruto (detrend)": sig, "Sinal filtrado": sig_f},
                                yaxis_title="Aceleração"),
        use_container_width=True,
    )

    # Densidade espectral de potência (Welch)
    nperseg = min(len(sig_f), int(fs * 4)) if fs > 0 else len(sig_f)
    nperseg = max(nperseg, 8)
    freqs, psd = welch(sig_f, fs=fs, nperseg=nperseg)

    band_mask = (freqs >= low) & (freqs <= high)
    if band_mask.sum() == 0:
        st.error("Faixa de frequência sem dados suficientes. Ajuste o filtro acima.")
        return

    peak_idx = np.argmax(psd[band_mask])
    peak_freq = float(freqs[band_mask][peak_idx])
    peak_power = float(psd[band_mask][peak_idx])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=psd, mode="lines", name="PSD"))
    fig.add_vline(x=peak_freq, line_dash="dash", line_color="#c00000",
                  annotation_text=f"{peak_freq:.2f} Hz", annotation_position="top")
    fig.update_layout(
        xaxis=dict(title="Frequência (Hz)", range=[0, min(20, fs / 2)],
                   showline=True, linecolor="black", mirror=True),
        yaxis=dict(title="Densidade espectral de potência", showline=True, linecolor="black", mirror=True),
    )
    st.plotly_chart(fig, use_container_width=True)

    rms_amplitude = float(np.sqrt(np.mean(sig_f ** 2)))
    # estimativa de amplitude de deslocamento a partir da aceleração senoidal:
    # amplitude_deslocamento ≈ amplitude_aceleração / (2π f)²
    omega = 2 * np.pi * peak_freq
    displacement_amplitude = rms_amplitude * np.sqrt(2) / (omega ** 2) if omega > 0 else np.nan

    resultados = {
        "Frequência de pico do tremor (Hz)": peak_freq,
        "Potência no pico": peak_power,
        "Amplitude RMS (aceleração filtrada)": rms_amplitude,
        "Amplitude de deslocamento estimada (mesma unidade²/s² → m)": displacement_amplitude,
    }
    st.markdown("#### Resultados")
    df = pd.DataFrame(resultados.items(), columns=["Métrica", "Valor"])
    df["Valor"] = df["Valor"].map(lambda v: f"{v:.4f}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if PARKINSON_BAND[0] <= peak_freq <= PARKINSON_BAND[1]:
        nota = f"Pico dentro da faixa clássica de tremor de repouso parkinsoniano ({PARKINSON_BAND[0]}–{PARKINSON_BAND[1]} Hz)."
    elif TREMOR_BAND[0] <= peak_freq <= TREMOR_BAND[1]:
        nota = "Pico dentro da faixa geral de tremor, fora da faixa típica parkinsoniana."
    else:
        nota = "Pico fora da faixa típica de tremor — verifique o posicionamento do sensor/ruído de movimento."
    st.info(nota)
    st.caption("Ferramenta de quantificação de sinal, sem finalidade diagnóstica isolada — interprete junto ao exame clínico.")
