"""
tug.py — Módulo Timed Up and Go (TUG) (Momentum Web)

Protocolo: celular na região lombar/tronco. O paciente levanta da cadeira,
caminha 3 m, vira, volta e senta. Arquivo CSV/TXT: Tempo, X, Y, Z
(acelerômetro e/ou giroscópio).

Como a colocação/orientação do sensor varia entre setups, a segmentação das
fases é semiautomática: sugerimos os instantes de transição a partir dos
picos do módulo do sinal (SVM) e o avaliador confirma/ajusta.
"""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import find_peaks

import common


PHASES = [
    ("t0", "Início (sentado)"),
    ("t1", "Fim de levantar / início da marcha de ida"),
    ("t2", "Início do giro (ida)"),
    ("t3", "Fim do giro / início da marcha de volta"),
    ("t4", "Início do giro final (volta)"),
    ("t5", "Fim do giro / início de sentar"),
    ("t6", "Fim (sentado)"),
]


def _suggest_times(t_sec: np.ndarray, svm: np.ndarray) -> dict:
    """Sugere 7 instantes-chave a partir dos picos do SVM, distribuídos no tempo."""
    if len(t_sec) < 10:
        return {key: float(t_sec[0]) for key, _ in PHASES}

    fs = common.sampling_rate(t_sec)
    distance = max(int(fs * 0.5), 1) if fs > 0 else 1
    peaks, props = find_peaks(svm, distance=distance, prominence=np.std(svm) * 0.3)

    duration = t_sec[-1] - t_sec[0]
    if len(peaks) >= 6:
        # usa os 6 picos mais proeminentes, ordenados no tempo, como transições
        top = peaks[np.argsort(props["prominences"])[-6:]]
        top = np.sort(top)
        times = [t_sec[0]] + [t_sec[i] for i in top] + [t_sec[-1]]
        # garante exatamente 7 valores
        times = sorted(set(times))
        while len(times) < 7:
            times.append(times[-1])
        times = times[:7]
    else:
        # fallback: divide a duração em 7 pontos igualmente espaçados
        times = list(t_sec[0] + duration * np.linspace(0, 1, 7))

    return {key: float(v) for (key, _), v in zip(PHASES, times)}


def render():
    st.subheader("🚶 Timed Up and Go (TUG)")
    st.caption(
        "Celular na região lombar/tronco durante o teste completo. "
        "Envie o arquivo CSV/TXT do sensor inercial (Tempo, X, Y, Z)."
    )

    data = common.get_uploaded_series("Selecione o arquivo CSV/TXT", "tug")
    if data is None:
        return

    t_sec, x, y, z = data
    svm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    svm_detrended = common.detrend(svm)

    st.plotly_chart(
        common.plot_timeseries(t_sec, {"X": x, "Y": y, "Z": z, "SVM": svm},
                                yaxis_title="Aceleração"),
        use_container_width=True,
    )

    state_key = "tug_times"
    if state_key not in st.session_state:
        st.session_state[state_key] = _suggest_times(t_sec, np.abs(svm_detrended))

    if st.button("🔄 Sugerir automaticamente as transições"):
        st.session_state[state_key] = _suggest_times(t_sec, np.abs(svm_detrended))
        st.rerun()

    st.markdown("#### Ajuste os instantes de cada fase (s)")
    tmin, tmax = float(t_sec[0]), float(t_sec[-1])
    cols = st.columns(2)
    new_times = {}
    for i, (key, label) in enumerate(PHASES):
        with cols[i % 2]:
            new_times[key] = st.number_input(
                label, min_value=tmin, max_value=tmax,
                value=float(np.clip(st.session_state[state_key][key], tmin, tmax)),
                step=0.05, key=f"tug_{key}",
            )
    st.session_state[state_key] = new_times

    markers = [{"x": v, "label": lbl.split(" ")[0], "color": "#c00000"}
               for (key, lbl), v in zip(PHASES, new_times.values())]
    st.plotly_chart(
        common.plot_timeseries(t_sec, {"SVM": svm}, yaxis_title="Aceleração (SVM)", markers=markers),
        use_container_width=True,
    )

    t0, t1, t2, t3, t4, t5, t6 = [new_times[k] for k, _ in PHASES]
    resultados = {
        "Tempo total do TUG (s)": t6 - t0,
        "Levantar da cadeira (s)": t1 - t0,
        "Marcha de ida (s)": t2 - t1,
        "Giro de ida (s)": t3 - t2,
        "Marcha de volta (s)": t4 - t3,
        "Giro final (s)": t5 - t4,
        "Sentar (s)": t6 - t5,
    }

    st.markdown("#### Resultados")
    df = pd.DataFrame(resultados.items(), columns=["Fase", "Duração (s)"])
    df["Duração (s)"] = df["Duração (s)"].map(lambda v: f"{v:.2f}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    total = resultados["Tempo total do TUG (s)"]
    if total < 10:
        risco = "Baixo risco de queda (referência: <10s)"
    elif total <= 20:
        risco = "Mobilidade normal a levemente reduzida (10–20s)"
    else:
        risco = "Risco aumentado de queda / mobilidade comprometida (>20s)"
    st.info(f"**Tempo total: {total:.2f} s** — {risco}")
    st.caption("Faixas de referência clássicas (Podsiadlo & Richardson, 1991) — uso orientativo, não diagnóstico.")
