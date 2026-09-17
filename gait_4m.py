"""
gait_4m.py — Módulo Caminhada de 4 metros (Momentum Web)

Protocolo: celular fixado no tronco/cintura. O paciente fica parado, anda
uma distância conhecida (padrão 4 m) e para. Arquivo: 4 colunas — Tempo
(ms) e Acelerômetro X, Y, Z (m/s²), separadas por ";" (formato exportado
pelo app, com cabeçalho em português: "Tempo (ms);Acc X (m/s²);...").

Método: detecta o início e o fim da caminhada a partir da envoltória (RMS
móvel) do componente dinâmico do sinal (removendo a gravidade/tendência),
comparando com o nível de repouso antes/depois. Dentro dessa janela, conta
os passos por detecção de picos na magnitude do sinal e calcula velocidade
de marcha, cadência e regularidade do passo.
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import find_peaks

import common


# =============================================================================
# Leitura do arquivo (Tempo(ms); Acc X; Acc Y; Acc Z — separado por ";")
# =============================================================================
@st.cache_data(show_spinner=False)
def _load_gait_df(raw: bytes) -> pd.DataFrame:
    text = common._decode_bytes(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    sep = ";" if text[:200].count(";") >= text[:200].count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", decimal=".")
    if df.shape[1] < 4:
        # tenta decimal com vírgula, caso o separador de coluna seja ";"
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", decimal=",")
    if df.shape[1] < 4:
        raise ValueError(f"O arquivo tem {df.shape[1]} coluna(s); preciso de pelo menos 4 (Tempo, X, Y, Z).")
    df4 = df.iloc[:, :4].copy()
    df4.columns = ["Tempo", "X", "Y", "Z"]
    for c in df4.columns:
        df4[c] = common.to_float_series(df4[c])
    df4 = df4.dropna().sort_values("Tempo").reset_index(drop=True)
    if df4.empty:
        raise ValueError("Nenhuma linha numérica válida encontrada.")
    return df4


def _get_uploaded_gait_series(label, key):
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
        df = _load_gait_df(raw)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None
    t = df["Tempo"].to_numpy(float)
    t_sec = t / 1000.0 if t.max() > 1000 else t  # heurística ms -> s
    return t_sec, df["X"].to_numpy(float), df["Y"].to_numpy(float), df["Z"].to_numpy(float)


# =============================================================================
# Detecção da janela de caminhada (início/fim) via envoltória RMS móvel
# =============================================================================
def _moving_rms(sig, win_samples):
    win_samples = max(1, int(win_samples))
    kernel = np.ones(win_samples) / win_samples
    return np.sqrt(np.convolve(sig ** 2, kernel, mode="same"))


def _detectar_janela_caminhada(t, svm_dinamico, fs, janela_s=0.5, k_limiar=3.0, dur_min_s=1.0):
    win_samples = max(1, int(janela_s * fs))
    envelope = _moving_rms(svm_dinamico, win_samples)

    n = len(t)
    borda = max(1, int(0.5 * fs))
    baseline_amostras = np.concatenate([envelope[:borda], envelope[-borda:]])
    baseline_mediana = float(np.median(baseline_amostras))
    baseline_std = float(np.std(baseline_amostras))
    limiar = baseline_mediana + k_limiar * max(baseline_std, 1e-3)

    acima = envelope > limiar
    idx = np.where(acima)[0]
    if idx.size == 0:
        return None, None, envelope, limiar

    # agrupa em segmentos contíguos e escolhe o mais longo com duração mínima
    segs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segs.append((start, prev))
            start = prev = i
    segs.append((start, prev))
    segs_validos = [s for s in segs if (t[s[1]] - t[s[0]]) >= dur_min_s]
    candidatos = segs_validos if segs_validos else segs
    s_best, e_best = max(candidatos, key=lambda se: t[se[1]] - t[se[0]])
    return int(s_best), int(e_best), envelope, limiar


# =============================================================================
# Detecção de passos (picos na magnitude, dentro da janela de caminhada)
# =============================================================================
def _detectar_passos(t, svm, i_ini, i_fim, fs, prominence_frac=0.3, min_step_time_s=0.25):
    sub_t = t[i_ini:i_fim + 1]
    sub_sig = svm[i_ini:i_fim + 1]
    if len(sub_sig) < 3:
        return np.array([], dtype=int)
    prom = prominence_frac * np.std(sub_sig)
    distance = max(1, int(min_step_time_s * fs))
    picos, _ = find_peaks(sub_sig, prominence=prom, distance=distance)
    return picos + i_ini


# =============================================================================
# Render
# =============================================================================
def render():
    st.subheader("🚶‍♂️ Caminhada de 4 metros")
    st.caption(
        "Celular fixado no tronco/cintura. O paciente fica parado, anda a distância marcada "
        "e para. Envie o arquivo com Tempo (ms) e Acelerômetro X, Y, Z."
    )

    data = _get_uploaded_gait_series("Selecione o arquivo (Tempo; Acc X; Acc Y; Acc Z)", "gait4m")
    if data is None:
        return
    t, x, y, z = data

    if len(t) < 20:
        st.error("Poucos pontos no arquivo para uma análise confiável.")
        return

    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt if dt > 0 else 0.0
    st.caption(f"Amostras: {len(t)} · Duração total do arquivo: {t[-1]-t[0]:.1f}s · Taxa de amostragem estimada: {fs:.1f} Hz")

    distancia_m = st.number_input("Distância percorrida (m)", min_value=1.0, max_value=50.0, value=4.0, step=0.5)

    svm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    svm_dinamico = common.detrend(svm)  # remove a gravidade/tendência, deixa só a oscilação da marcha

    # ---- detecção do início/fim da caminhada ----
    st.markdown("#### 🔍 Detecção do início e fim da caminhada")
    c1, c2, c3 = st.columns(3)
    with c1:
        janela_s = st.number_input("Janela do envelope (s)", min_value=0.1, max_value=2.0, value=0.5, step=0.1, key="g4m_win")
    with c2:
        k_limiar = st.number_input("Sensibilidade do limiar (× DP repouso)", min_value=1.0, max_value=10.0, value=3.0, step=0.5, key="g4m_k")
    with c3:
        dur_min_s = st.number_input("Duração mínima da caminhada (s)", min_value=0.2, max_value=5.0, value=1.0, step=0.1, key="g4m_durmin")

    i_ini, i_fim, envelope, limiar = _detectar_janela_caminhada(t, svm_dinamico, fs, janela_s, k_limiar, dur_min_s)

    if i_ini is None:
        st.error("Não foi possível identificar o início/fim da caminhada automaticamente. Ajuste a sensibilidade acima.")
        fig0 = common.plot_timeseries(t, {"Envelope RMS": envelope}, yaxis_title="m/s²")
        st.plotly_chart(fig0, use_container_width=True)
        return

    tmin_s, tmax_s = float(t[0]), float(t[-1])
    c4, c5 = st.columns(2)
    with c4:
        t_ini_ajustado = st.number_input("Início da caminhada (s)", min_value=tmin_s, max_value=tmax_s,
                                          value=float(t[i_ini]), step=0.05, key="g4m_t_ini")
    with c5:
        t_fim_ajustado = st.number_input("Fim da caminhada (s)", min_value=tmin_s, max_value=tmax_s,
                                          value=float(t[i_fim]), step=0.05, key="g4m_t_fim")
    i_ini = int(np.argmin(np.abs(t - t_ini_ajustado)))
    i_fim = int(np.argmin(np.abs(t - t_fim_ajustado)))
    if i_fim <= i_ini:
        st.error("O fim precisa ser depois do início.")
        return

    # ---- detecção de passos ----
    st.markdown("#### 🔍 Detecção de passos")
    c6, c7 = st.columns(2)
    with c6:
        prominence_frac = st.slider("Sensibilidade do pico (× DP do sinal)", 0.05, 1.0, 0.3, 0.05, key="g4m_prom")
    with c7:
        min_step_time = st.number_input("Tempo mínimo entre passos (s)", min_value=0.15, max_value=1.0, value=0.40, step=0.05, key="g4m_minstep")

    idx_passos = _detectar_passos(t, svm, i_ini, i_fim, fs, prominence_frac, min_step_time)

    # ---- gráfico ----
    st.markdown("#### Sinal e passos detectados")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=svm, mode="lines", name="Magnitude (SVM)", line=dict(color="black", width=1)))
    fig.add_vrect(x0=t[i_ini], x1=t[i_fim], fillcolor="rgba(31,74,255,0.08)", line_width=0,
                  annotation_text="Caminhada", annotation_position="top left")
    if idx_passos.size:
        fig.add_trace(go.Scatter(x=t[idx_passos], y=svm[idx_passos], mode="markers", name="Passos",
                                  marker=dict(color="crimson", size=7, symbol="diamond")))
    fig.update_layout(xaxis_title="Tempo (s)", yaxis_title="Aceleração (m/s²)", height=420,
                       plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)

    # ---- variáveis ----
    duracao = float(t[i_fim] - t[i_ini])
    n_passos = int(idx_passos.size)
    velocidade = distancia_m / duracao if duracao > 0 else np.nan
    cadencia = (n_passos / duracao * 60.0) if duracao > 0 else np.nan

    tempos_passo = np.diff(t[idx_passos]) if n_passos > 1 else np.array([])
    tempo_passo_medio = float(np.mean(tempos_passo)) if tempos_passo.size else np.nan
    tempo_passo_sd = float(np.std(tempos_passo, ddof=1)) if tempos_passo.size > 1 else np.nan
    tempo_passo_cv = (tempo_passo_sd / tempo_passo_medio * 100.0) if (tempos_passo.size > 1 and tempo_passo_medio) else np.nan
    comprimento_passo_medio = (distancia_m / n_passos) if n_passos > 0 else np.nan

    st.markdown("#### Resultados")
    resultados = {
        "Duração da caminhada (s)": duracao,
        "Distância (m)": distancia_m,
        "Velocidade de marcha (m/s)": velocidade,
        "Número de passos": n_passos,
        "Cadência (passos/min)": cadencia,
        "Comprimento médio do passo (m)": comprimento_passo_medio,
        "Tempo médio de passo (s)": tempo_passo_medio,
        "DP do tempo de passo (s)": tempo_passo_sd,
        "CV do tempo de passo (%)": tempo_passo_cv,
    }
    df_res = pd.DataFrame(resultados.items(), columns=["Variável", "Valor"])
    df_res["Valor"] = df_res["Valor"].map(
        lambda v: f"{v:.3f}" if isinstance(v, float) and not np.isnan(v) else (str(v) if isinstance(v, int) else "—")
    )
    st.dataframe(df_res, use_container_width=True, hide_index=True)

    st.info(f"**Velocidade de marcha: {velocidade:.2f} m/s** ({n_passos} passos em {duracao:.2f} s)")
    st.caption(
        "Velocidade de marcha < 0,8 m/s é um marcador clássico de risco de fragilidade/desfechos "
        "adversos em idosos (uso orientativo, não diagnóstico). Ajuste os pontos de início/fim e a "
        "sensibilidade de detecção de passos acima se necessário."
    )
