"""
joint_position.py — Módulo Posicionamento Articular / Senso de Posição
Articular (JPS) (Momentum Web)

Protocolo: celular fixado ao segmento avaliado (ex.: face anterior da tíbia
para joelho). O avaliador posiciona o segmento no ângulo-alvo (o paciente
memoriza), retorna à posição neutra e, de olhos fechados, tenta reproduzir o
ângulo-alvo — geralmente repetido 3 a 5 vezes. Arquivo CSV/TXT: Tempo, X, Y, Z
(acelerômetro e/ou giroscópio), gravado continuamente durante todo o teste.
"""
import numpy as np
import pandas as pd
import streamlit as st

import common


def _accel_tilt_angle(x, y, z, eixo_referencia: str):
    """Ângulo de inclinação (graus) a partir do acelerômetro (não integra, sem deriva)."""
    if eixo_referencia == "X (flexão em torno de X)":
        return np.degrees(np.arctan2(x, np.sqrt(y ** 2 + z ** 2)))
    if eixo_referencia == "Y (flexão em torno de Y)":
        return np.degrees(np.arctan2(y, np.sqrt(x ** 2 + z ** 2)))
    return np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))


def _gyro_integrated_angle(t_sec, g_sig):
    """Ângulo (graus) por integração do giroscópio (sujeito a deriva)."""
    dt = np.diff(t_sec, prepend=t_sec[0])
    return np.cumsum(g_sig * dt)


def render():
    st.subheader("🦵 Posicionamento Articular (Senso de Posição Articular)")
    st.caption(
        "Sensor fixado ao segmento avaliado (ex.: tíbia para joelho). Grave o teste completo: "
        "posicionamento no ângulo-alvo, retorno à posição neutra e cada tentativa de reprodução."
    )

    fonte = st.radio("Fonte do ângulo", ["Acelerômetro (inclinação, sem deriva)", "Giroscópio (integração angular)"])

    data = common.get_uploaded_series("Selecione o arquivo CSV/TXT", "joint_position")
    if data is None:
        return

    t_sec, x, y, z = data

    if fonte.startswith("Acelerômetro"):
        eixo_ref = st.selectbox("Eixo de referência do movimento", ["X (flexão em torno de X)", "Y (flexão em torno de Y)", "Z (flexão em torno de Z)"])
        angle = _accel_tilt_angle(x, y, z, eixo_ref)
    else:
        eixo_g = st.selectbox("Eixo do giroscópio correspondente ao movimento", ["X", "Y", "Z"])
        g_sig = {"X": x, "Y": y, "Z": z}[eixo_g]
        angle = _gyro_integrated_angle(t_sec, g_sig)
        st.warning("Integração do giroscópio acumula deriva ao longo do tempo — prefira janelas de tempo curtas por tentativa.")

    st.plotly_chart(
        common.plot_timeseries(t_sec, {"Ângulo": angle}, yaxis_title="Ângulo (graus)"),
        use_container_width=True,
    )

    n_trials = st.number_input("Número de tentativas de reprodução", min_value=1, max_value=8, value=3, step=1)

    tmin, tmax = float(t_sec[0]), float(t_sec[-1])
    st.markdown("#### Janela do ângulo-alvo (posição apresentada pelo avaliador)")
    c1, c2 = st.columns(2)
    with c1:
        target_start = st.number_input("Início (s)", min_value=tmin, max_value=tmax, value=tmin, step=0.1, key="jps_target_start")
    with c2:
        target_end = st.number_input("Fim (s)", min_value=tmin, max_value=tmax, value=min(tmin + 2.0, tmax), step=0.1, key="jps_target_end")

    target_mask = (t_sec >= target_start) & (t_sec <= target_end)
    target_angle = float(np.mean(angle[target_mask])) if target_mask.sum() > 0 else np.nan
    st.metric("Ângulo-alvo médio (graus)", f"{target_angle:.2f}" if not np.isnan(target_angle) else "—")

    st.markdown("#### Janelas de cada tentativa de reprodução")
    trial_results = []
    for i in range(int(n_trials)):
        c1, c2 = st.columns(2)
        with c1:
            ts = st.number_input(f"Tentativa {i+1} — início (s)", min_value=tmin, max_value=tmax,
                                  value=min(tmin + 3.0 + 3.0 * i, tmax), step=0.1, key=f"jps_trial_start_{i}")
        with c2:
            te = st.number_input(f"Tentativa {i+1} — fim (s)", min_value=tmin, max_value=tmax,
                                  value=min(tmin + 5.0 + 3.0 * i, tmax), step=0.1, key=f"jps_trial_end_{i}")
        mask = (t_sec >= ts) & (t_sec <= te)
        reproduced = float(np.mean(angle[mask])) if mask.sum() > 0 else np.nan
        error = reproduced - target_angle if not np.isnan(reproduced) and not np.isnan(target_angle) else np.nan
        trial_results.append({
            "Tentativa": i + 1,
            "Ângulo alvo (°)": target_angle,
            "Ângulo reproduzido (°)": reproduced,
            "Erro (°)": error,
            "Erro absoluto (°)": abs(error) if not np.isnan(error) else np.nan,
        })

    df = pd.DataFrame(trial_results)
    st.dataframe(df.style.format({c: "{:.2f}" for c in df.columns if c != "Tentativa"}),
                 use_container_width=True, hide_index=True)

    errors = df["Erro (°)"].dropna().to_numpy()
    abs_errors = df["Erro absoluto (°)"].dropna().to_numpy()
    if len(errors) > 0:
        st.markdown("#### Resumo — Métricas clássicas de JPS")
        resumo = {
            "Erro Absoluto médio (AE) (°)": float(np.mean(abs_errors)),
            "Erro Constante (CE) — viés (°)": float(np.mean(errors)),
            "Erro Variável (VE) — desvio-padrão (°)": float(np.std(errors, ddof=1)) if len(errors) > 1 else np.nan,
        }
        df_resumo = pd.DataFrame(resumo.items(), columns=["Métrica", "Valor"])
        df_resumo["Valor"] = df_resumo["Valor"].map(lambda v: f"{v:.2f}" if not np.isnan(v) else "—")
        st.dataframe(df_resumo, use_container_width=True, hide_index=True)
        st.caption(
            "AE: precisão geral (quanto menor, melhor). CE: tendência a superestimar (+) ou subestimar (−) o ângulo. "
            "VE: consistência entre tentativas (quanto menor, mais consistente)."
        )
