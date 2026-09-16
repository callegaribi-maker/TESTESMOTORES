"""
tug.py — Módulo Timed Up and Go (TUG) (Momentum Web)

Porta o pipeline do MomentumTUG (protocolo-tug-servidor-smartphone.md):
captura de acelerômetro + giroscópio, pré-processamento (detrend → magnitude
→ interpolação 100Hz → filtro RC passa-baixa de 1ª ordem), segmentação do
giroscópio por k-means 1D (7 estados) com regra de estabilidade, e
identificação dos pontos-chave (Início, Fim, G1, G2, A1, A2).

⚠️ A fórmula EXATA de identificação dos pontos-chave está documentada em
`docs/calculo-pontos-chave.md` no projeto original (não disponível aqui). A
implementação abaixo é uma aproximação de boa-fé a partir das regras e
constantes que ESTÃO documentadas no protocolo (janela A1/A2 = 1250ms,
distância mínima G1/G2 = 1000ms, Início só ≥ 4000ms, lookback 4/9 amostras).
Os pontos-chave são editáveis manualmente na tela — confira sempre contra o
gráfico antes de usar o resultado.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import find_peaks

import common

# ---- Constantes fixas do pipeline (seção 2 do protocolo) ----
FS_TARGET = 100.0            # Hz, interpolação
GYRO_CUTOFF = 1.5            # Hz
ACEL_CUTOFF = 6.0            # Hz
KMEANS_STATES = 7
STABILITY_0 = 10             # amostras p/ "vencer" e virar estado 0 (repouso)
STABILITY_OTHER = 5          # amostras p/ vencer qualquer outro estado
LOOKBACK_START_SAMPLES = 4   # 40 ms a 100 Hz
LOOKBACK_END_SAMPLES = 9     # 90 ms a 100 Hz
INICIO_MIN_MS = 4000.0       # Início só pode ser marcado a partir daqui
A1A2_WINDOW_MS = 1250.0
MIN_PEAK_DISTANCE_G1G2_MS = 1000.0


# =============================================================================
# Upload (preserva Tempo em ms — os cálculos do protocolo são todos em ms)
# =============================================================================
def _load_ms_series(label, key):
    uploaded = st.file_uploader(label, type=["csv", "txt"], key=f"{key}_uploader")
    raw_key = f"{key}_raw"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = None
    if uploaded is not None:
        st.session_state[raw_key] = uploaded.getvalue()
    raw = st.session_state[raw_key]
    if not raw:
        return None
    try:
        df = common.load_first4cols_cached(raw)
    except Exception as e:
        st.error(f"Erro ao ler '{label}': {e}")
        return None
    t = common.to_float_series(df["Tempo"]).to_numpy(float)
    x = common.to_float_series(df["X"]).to_numpy(float)
    y = common.to_float_series(df["Y"]).to_numpy(float)
    z = common.to_float_series(df["Z"]).to_numpy(float)
    valid = ~np.isnan(t) & ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    t, x, y, z = t[valid], x[valid], y[valid], z[valid]
    if len(t) < 5:
        st.error(f"Poucos pontos válidos em '{label}'.")
        return None
    order = np.argsort(t)
    t, x, y, z = t[order], x[order], y[order], z[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], x[keep], y[keep], z[keep]  # t em ms


# =============================================================================
# Pipeline de pré-processamento (seção 2.1)
# =============================================================================
def _detrend_mean(sig):
    return sig - np.mean(sig)


def _interpolate_100hz(t_ms, sig):
    t0, t1 = t_ms[0], t_ms[-1]
    step = 1000.0 / FS_TARGET
    n = int(np.floor((t1 - t0) / step)) + 1
    t_new = t0 + np.arange(n) * step
    sig_new = np.interp(t_new, t_ms, sig)
    return t_new, sig_new


def _rc_lowpass(sig, fs, cutoff_hz):
    """Filtro RC passa-baixa de 1ª ordem (não zero-fase), igual ao app original."""
    dt = 1.0 / fs
    rc = 1.0 / (2 * np.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    out = np.empty_like(sig, dtype=float)
    out[0] = sig[0]
    for i in range(1, len(sig)):
        out[i] = out[i - 1] + alpha * (sig[i] - out[i - 1])
    return out


def _process_signal(t_ms, x, y, z, cutoff_hz):
    """detrend (média) -> magnitude -> interpolação 100Hz -> filtro RC passa-baixa."""
    xd, yd, zd = _detrend_mean(x), _detrend_mean(y), _detrend_mean(z)
    mag = np.sqrt(xd ** 2 + yd ** 2 + zd ** 2)
    t100, mag100 = _interpolate_100hz(t_ms, mag)
    mag_f = _rc_lowpass(mag100, FS_TARGET, cutoff_hz)
    return t100, mag_f


# =============================================================================
# Segmentação do giroscópio (seção 2.2)
# =============================================================================
def _kmeans_1d(sig, k=KMEANS_STATES, iters=10):
    centroids = np.linspace(sig.min(), sig.max(), k)
    states = np.zeros(len(sig), dtype=int)
    for _ in range(iters):
        dist = np.abs(sig[:, None] - centroids[None, :])
        states = np.argmin(dist, axis=1)
        for c in range(k):
            mask = states == c
            if mask.any():
                centroids[c] = sig[mask].mean()
    dist = np.abs(sig[:, None] - centroids[None, :])
    states = np.argmin(dist, axis=1)
    order = np.argsort(centroids)          # estado 0 = menor magnitude (repouso)
    remap = {old: new for new, old in enumerate(order)}
    return np.array([remap[s] for s in states])


def _apply_stability_rule(states, stability0=STABILITY_0, stability_other=STABILITY_OTHER):
    n = len(states)
    stable = np.zeros(n, dtype=int)
    cur_stable = states[0]
    run_state = states[0]
    run_len = 1
    stable[0] = cur_stable
    for i in range(1, n):
        if states[i] == run_state:
            run_len += 1
        else:
            run_state = states[i]
            run_len = 1
        threshold = stability0 if run_state == 0 else stability_other
        if run_state != cur_stable and run_len >= threshold:
            cur_stable = run_state
        stable[i] = cur_stable
    return stable


# =============================================================================
# Pontos-chave (aproximação — ver aviso no topo do arquivo)
# =============================================================================
def _find_inicio_fim(t_ms, stable_states):
    valid_idx = np.where(t_ms >= INICIO_MIN_MS)[0]
    if valid_idx.size == 0:
        return None, None
    moving_idx = valid_idx[stable_states[valid_idx] != 0]
    if moving_idx.size == 0:
        return None, None
    i_inicio = max(int(moving_idx[0]) - LOOKBACK_START_SAMPLES, 0)
    i_fim = max(int(moving_idx[-1]) - LOOKBACK_END_SAMPLES, i_inicio)
    return i_inicio, i_fim


def _find_g1_g2(mag_f, i_inicio, i_fim, fs=FS_TARGET):
    sub_sig = mag_f[i_inicio:i_fim + 1]
    if len(sub_sig) < 3:
        return None, None
    distance = max(int(MIN_PEAK_DISTANCE_G1G2_MS / 1000.0 * fs), 1)
    peaks, props = find_peaks(sub_sig, distance=distance, prominence=np.std(sub_sig) * 0.2)
    if len(peaks) == 0:
        return None, None
    if len(peaks) == 1:
        return int(peaks[0] + i_inicio), None
    top2 = peaks[np.argsort(props["prominences"])[-2:]]
    top2 = np.sort(top2) + i_inicio
    return int(top2[0]), int(top2[1])


def _find_peak_in_window(t_ms, mag_f, t_start_ms, t_end_ms):
    mask = (t_ms >= t_start_ms) & (t_ms <= t_end_ms)
    if not mask.any():
        return None
    idx_local = int(np.argmax(mag_f[mask]))
    return int(np.where(mask)[0][idx_local])


def _nearest_index(t_arr, target_ms):
    return int(np.argmin(np.abs(t_arr - target_ms)))


def _validate(stand, sit, walk_out, walk_back):
    if any(v is None or np.isnan(v) for v in [stand, sit, walk_out, walk_back]):
        return None, "Não foi possível validar o teste."
    ok = True
    if not (stand < walk_out and stand < walk_back):
        ok = False
    if not (sit < walk_out and sit < walk_back):
        ok = False
    longest, shortest = max(walk_out, walk_back), min(walk_out, walk_back)
    if shortest <= 0 or (longest / shortest) >= 2.5:
        ok = False
    msg = "" if ok else "Marcação incorreta dos pontos-chave — confira o gráfico e ajuste manualmente."
    return ok, msg


# =============================================================================
# Render
# =============================================================================
def render():
    st.subheader("🚶 Timed Up and Go (TUG)")
    st.caption(
        "Protocolo MomentumTUG: acelerômetro + giroscópio na região lombar/tronco, "
        "com beep de referência em 5000ms. Envie os DOIS arquivos abaixo (Tempo, X, Y, Z)."
    )
    st.warning(
        "A fórmula exata dos pontos-chave (docs/calculo-pontos-chave.md) não estava disponível — "
        "esta é uma aproximação a partir do pipeline documentado (k-means + regra de estabilidade "
        "+ janelas de 1250/1000/4000ms). Os pontos são **editáveis manualmente** abaixo — "
        "sempre confira contra o gráfico antes de usar o resultado.",
        icon="⚠️",
    )

    col_u1, col_u2 = st.columns(2)
    with col_u1:
        accel_data = _load_ms_series("📥 Arquivo do ACELERÔMETRO (Tempo,X,Y,Z — ms / m·s⁻²)", "tug_accel")
    with col_u2:
        gyro_data = _load_ms_series("📥 Arquivo do GIROSCÓPIO (Tempo,X,Y,Z — ms / rad·s⁻¹)", "tug_gyro")

    if accel_data is None or gyro_data is None:
        st.info("Envie os dois arquivos (acelerômetro e giroscópio) para iniciar a análise.")
        return

    t_a_ms, xa, ya, za = accel_data
    t_g_ms, xg, yg, zg = gyro_data

    # ---- pipeline ----
    t_g100, gyro_mag_f = _process_signal(t_g_ms, xg, yg, zg, GYRO_CUTOFF)
    t_a100, acel_mag_f = _process_signal(t_a_ms, xa, ya, za, ACEL_CUTOFF)

    # ---- segmentação do giroscópio ----
    raw_states = _kmeans_1d(gyro_mag_f)
    stable_states = _apply_stability_rule(raw_states)

    # ---- pontos-chave (detecção automática) ----
    i_inicio, i_fim = _find_inicio_fim(t_g100, stable_states)
    if i_inicio is None or i_fim is None:
        st.error(
            "Não foi possível identificar Início/Fim automaticamente (nenhuma amostra ≥ 4000ms "
            "com estado de movimento detectado). Confira o arquivo do giroscópio."
        )
        st.plotly_chart(
            common.plot_timeseries(t_g100 / 1000.0, {"Giroscópio |mag| filtrado": gyro_mag_f},
                                    yaxis_title="rad/s"),
            use_container_width=True,
        )
        return

    t_inicio_auto, t_fim_auto = t_g100[i_inicio], t_g100[i_fim]

    i_g1, i_g2 = _find_g1_g2(gyro_mag_f, i_inicio, i_fim)
    t_g1_auto = t_g100[i_g1] if i_g1 is not None else t_inicio_auto + (t_fim_auto - t_inicio_auto) / 3
    t_g2_auto = t_g100[i_g2] if i_g2 is not None else t_inicio_auto + 2 * (t_fim_auto - t_inicio_auto) / 3

    a1_end_ms = min(t_inicio_auto + A1A2_WINDOW_MS, t_fim_auto)
    a2_start_ms = max(t_inicio_auto, t_fim_auto - A1A2_WINDOW_MS)
    i_a1 = _find_peak_in_window(t_a100, acel_mag_f, t_inicio_auto, a1_end_ms)
    i_a2 = _find_peak_in_window(t_a100, acel_mag_f, a2_start_ms, t_fim_auto)
    t_a1_auto = t_a100[i_a1] if i_a1 is not None else t_inicio_auto
    t_a2_auto = t_a100[i_a2] if i_a2 is not None else t_fim_auto

    if i_g1 is None or i_g2 is None or i_a1 is None or i_a2 is None:
        st.warning("Alguns pontos-chave não foram detectados automaticamente com confiança — confira e ajuste manualmente abaixo.")

    # ---- ajuste manual (fonte da verdade para os cálculos) ----
    st.markdown("#### Ajuste fino dos pontos-chave (ms, a partir do início da captura)")
    tmin_ms = float(min(t_g100[0], t_a100[0]))
    tmax_ms = float(max(t_g100[-1], t_a100[-1]))

    def _num(label, val, key):
        return st.number_input(label, min_value=tmin_ms, max_value=tmax_ms,
                                value=float(np.clip(val, tmin_ms, tmax_ms)),
                                step=10.0, key=key)

    c1, c2, c3 = st.columns(3)
    with c1:
        t_inicio_ms = _num("Início (ms)", t_inicio_auto, "tug_t_inicio")
        t_a1_ms = _num("A1 — pico levantar (ms)", t_a1_auto, "tug_t_a1")
    with c2:
        t_g1_ms = _num("G1 — giro de ida (ms)", t_g1_auto, "tug_t_g1")
        t_g2_ms = _num("G2 — giro de volta (ms)", t_g2_auto, "tug_t_g2")
    with c3:
        t_a2_ms = _num("A2 — pico sentar (ms)", t_a2_auto, "tug_t_a2")
        t_fim_ms = _num("Fim (ms)", t_fim_auto, "tug_t_fim")

    # recalcula os índices/picos a partir dos valores finais (auto OU editado à mão)
    i_g1_f = _nearest_index(t_g100, t_g1_ms)
    i_g2_f = _nearest_index(t_g100, t_g2_ms)
    i_a1_f = _nearest_index(t_a100, t_a1_ms)
    i_a2_f = _nearest_index(t_a100, t_a2_ms)
    pico_g1, pico_g2 = float(gyro_mag_f[i_g1_f]), float(gyro_mag_f[i_g2_f])
    pico_a1, pico_a2 = float(acel_mag_f[i_a1_f]), float(acel_mag_f[i_a2_f])

    # ---- gráficos (seção 4) ----
    st.markdown("#### Giroscópio")
    fig_g = go.Figure()
    fig_g.add_trace(go.Scatter(x=t_g100 / 1000.0, y=gyro_mag_f, mode="lines",
                                name="Vel. Angular", line=dict(color="black", width=1.5)))
    for label, tm, color in [("Início", t_inicio_ms, "red"), ("Fim", t_fim_ms, "blue"),
                              ("G1", t_g1_ms, "#FBC02D"), ("G2", t_g2_ms, "#2E7D32")]:
        fig_g.add_vline(x=tm / 1000.0, line_dash="dash", line_color=color,
                         annotation_text=label, annotation_position="top")
    fig_g.update_layout(xaxis_title="Tempo (s)", yaxis_title="Vel. Angular (rad/s)",
                         plot_bgcolor="white", paper_bgcolor="white")
    fig_g.update_xaxes(showgrid=True, gridcolor="#eee")
    fig_g.update_yaxes(showgrid=True, gridcolor="#eee")
    st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("#### Acelerômetro")
    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(x=t_a100 / 1000.0, y=acel_mag_f, mode="lines",
                                name="Aceleração", line=dict(color="black", width=1.5)))
    for label, tm, color in [("Início", t_inicio_ms, "red"), ("Fim", t_fim_ms, "blue"),
                              ("A1", t_a1_ms, "#FBC02D"), ("A2", t_a2_ms, "#2E7D32")]:
        fig_a.add_vline(x=tm / 1000.0, line_dash="dash", line_color=color,
                         annotation_text=label, annotation_position="top")
    fig_a.update_layout(xaxis_title="Tempo (s)", yaxis_title="Aceleração (m/s²)",
                         plot_bgcolor="white", paper_bgcolor="white")
    fig_a.update_xaxes(showgrid=True, gridcolor="#eee")
    fig_a.update_yaxes(showgrid=True, gridcolor="#eee")
    st.plotly_chart(fig_a, use_container_width=True)

    # ---- relatório (seção 3.1) ----
    duracao_total = (t_fim_ms - t_inicio_ms) / 1000.0
    duracao_levantar = (t_a1_ms - t_inicio_ms) / 1000.0
    duracao_ida = (t_g1_ms - t_a1_ms) / 1000.0
    duracao_volta = (t_a2_ms - t_g1_ms) / 1000.0
    duracao_sentar = (t_fim_ms - t_a2_ms) / 1000.0

    st.markdown("#### Relatório")
    resultados = {
        "Duração total (s)": duracao_total,
        "Duração do levantar-se (s)": duracao_levantar,
        "Duração da caminhada de ida (s)": duracao_ida,
        "Duração da caminhada de volta (s)": duracao_volta,
        "Duração do sentar-se (s)": duracao_sentar,
        "Pico G1 (rad/s)": pico_g1,
        "Pico G2 (rad/s)": pico_g2,
        "Pico A1 (m/s²)": pico_a1,
        "Pico A2 (m/s²)": pico_a2,
    }
    df = pd.DataFrame(resultados.items(), columns=["Campo", "Valor"])
    casas = {"Duração total (s)": 3, "Duração do levantar-se (s)": 3, "Duração da caminhada de ida (s)": 3,
             "Duração da caminhada de volta (s)": 3, "Duração do sentar-se (s)": 3,
             "Pico G1 (rad/s)": 4, "Pico G2 (rad/s)": 4, "Pico A1 (m/s²)": 4, "Pico A2 (m/s²)": 4}
    df["Valor"] = [f"{v:.{casas[k]}f}" for k, v in resultados.items()]
    st.dataframe(df, use_container_width=True, hide_index=True)

    validado, msg_validacao = _validate(duracao_levantar, duracao_sentar, duracao_ida, duracao_volta)
    if validado is None:
        st.info(msg_validacao)
    elif validado:
        st.success(f"✅ Teste válido. Duração total: {duracao_total:.2f} s")
    else:
        st.warning(f"⚠️ {msg_validacao} (Duração total: {duracao_total:.2f} s)")

    st.caption(
        "Regras de validação: levantar-se e sentar-se devem durar menos que cada trecho de caminhada; "
        "a razão entre a caminhada mais longa e a mais curta deve ser < 2,5."
    )
