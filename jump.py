"""
jump.py — Módulo Salto Vertical (Momentum Web)

Protocolo: celular fixado no tronco/cintura (próximo ao centro de massa),
gravação contínua do acelerômetro vertical durante um salto (contramovimento
→ propulsão → voo → aterrissagem). Arquivo: 2 colunas — Tempo (s) e
Aceleração vertical (m/s²) — com ou sem cabeçalho, separadas por espaço,
tab, vírgula ou ponto-e-vírgula (ex.: "Tempo(s)  Aceleracao_Vertical(m/s2)").

Método: tempo de voo (flight-time method) — identifica a janela onde a
aceleração fica próxima de zero (queda livre = sem força de contato) para
marcar decolagem e aterrissagem, e calcula a altura do salto a partir dessa
duração.
"""
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import common

G = 9.81  # m/s²


# =============================================================================
# Leitura do arquivo (2 colunas: Tempo, Aceleração vertical)
# =============================================================================
def _tem_cabecalho(primeira_linha, sep_regex=r"[\s,;]+"):
    partes = [p for p in re.split(sep_regex, primeira_linha.strip()) if p]
    for p in partes:
        try:
            float(p.replace(",", "."))
        except ValueError:
            return True
    return False


@st.cache_data(show_spinner=False)
def _load_jump_df(raw: bytes) -> pd.DataFrame:
    text = common._decode_bytes(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")  # normaliza quebras de linha (inclui \r solto, estilo Mac antigo)
    linhas = [l for l in text.splitlines() if l.strip()]
    if not linhas:
        raise ValueError("Arquivo vazio.")
    tem_header = _tem_cabecalho(linhas[0])

    df = pd.read_csv(
        io.StringIO(text), sep=r"[\s,;]+", engine="python",
        header=0 if tem_header else None,
    )
    if df.shape[1] < 2:
        raise ValueError(f"O arquivo tem {df.shape[1]} coluna(s); preciso de pelo menos 2 (Tempo, Aceleração).")

    df2 = df.iloc[:, :2].copy()
    df2.columns = ["Tempo", "Aceleracao"]
    df2["Tempo"] = common.to_float_series(df2["Tempo"])
    df2["Aceleracao"] = common.to_float_series(df2["Aceleracao"])
    df2 = df2.dropna()
    if df2.empty:
        raise ValueError("Nenhuma linha numérica válida encontrada nas 2 primeiras colunas.")
    df2 = df2.sort_values("Tempo").reset_index(drop=True)
    return df2


def _get_uploaded_jump_series(label, key):
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
        df = _load_jump_df(raw)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        return None
    return df["Tempo"].to_numpy(float), df["Aceleracao"].to_numpy(float)


# =============================================================================
# Detecção da fase de voo e variáveis do salto
# =============================================================================
def _segmentos_contiguos(idx):
    if len(idx) == 0:
        return []
    segs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            segs.append((start, prev))
            start = prev = i
    segs.append((start, prev))
    return segs


def _detectar_voo(t, a, limiar=2.0, dur_min=0.0):
    mask = np.abs(a) < limiar
    idx = np.where(mask)[0]
    segs = _segmentos_contiguos(idx)
    if not segs:
        return None
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    segs_validos = [s for s in segs if (t[s[1]] - t[s[0]] + dt) >= dur_min]
    candidatos = segs_validos if segs_validos else segs
    s_best, e_best = max(candidatos, key=lambda se: t[se[1]] - t[se[0]])
    return s_best, e_best


def _calcular_variaveis(t, a, s_voo, e_voo):
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    n_amostras_voo = e_voo - s_voo + 1
    tempo_voo = n_amostras_voo * dt

    altura_m = G * tempo_voo ** 2 / 8.0
    v_decolagem = G * tempo_voo / 2.0

    pre = a[:s_voo] if s_voo > 0 else np.array([])
    pos = a[e_voo + 1:] if e_voo + 1 < len(a) else np.array([])

    pico_propulsao = float(np.max(pre)) if pre.size else float("nan")
    idx_pico_propulsao = int(np.argmax(pre)) if pre.size else None
    contramovimento_min = float(np.min(pre[:idx_pico_propulsao])) if (pre.size and idx_pico_propulsao and idx_pico_propulsao > 0) else (float(np.min(pre)) if pre.size else float("nan"))
    idx_contramovimento = int(np.argmin(pre[:idx_pico_propulsao])) if (pre.size and idx_pico_propulsao and idx_pico_propulsao > 0) else (int(np.argmin(pre)) if pre.size else None)

    pico_pouso = float(np.max(pos)) if pos.size else float("nan")
    idx_pico_pouso = int(np.argmax(pos)) if pos.size else None

    return {
        "t_decolagem": float(t[s_voo]),
        "t_aterrissagem": float(t[e_voo]) + dt,
        "tempo_voo": tempo_voo,
        "altura_salto_cm": altura_m * 100.0,
        "v_decolagem": v_decolagem,
        "pico_propulsao": pico_propulsao,
        "t_pico_propulsao": float(t[idx_pico_propulsao]) if idx_pico_propulsao is not None else None,
        "contramovimento_min": contramovimento_min,
        "t_contramovimento": float(t[idx_contramovimento]) if idx_contramovimento is not None else None,
        "pico_pouso": pico_pouso,
        "t_pico_pouso": float(t[e_voo + 1 + idx_pico_pouso]) if idx_pico_pouso is not None else None,
    }


# =============================================================================
# Render
# =============================================================================
def render():
    st.subheader("🦘 Salto Vertical")
    st.caption(
        "Celular fixado no tronco/cintura, próximo ao centro de massa. Envie um arquivo "
        "com 2 colunas — Tempo (s) e Aceleração vertical (m/s²) — com ou sem cabeçalho."
    )

    data = _get_uploaded_jump_series("Selecione o arquivo (Tempo, Aceleração vertical)", "jump")
    if data is None:
        return
    t, a = data

    if len(t) < 5:
        st.error("Poucos pontos no arquivo para uma análise confiável.")
        return

    dt = float(np.median(np.diff(t)))
    fs = 1.0 / dt if dt > 0 else 0.0
    st.caption(f"Amostras: {len(t)} · Taxa de amostragem estimada: {fs:.1f} Hz")

    # ---- filtro passa-baixa (opcional) ----
    st.markdown("#### 🎚️ Filtro passa-baixa (opcional)")
    c_filt, c_cut = st.columns(2)
    with c_filt:
        aplicar_filtro = st.checkbox("Aplicar filtro passa-baixa", value=False, key="jump_apply_filter")
    with c_cut:
        cutoff_hz = st.number_input(
            "Frequência de corte (Hz)", min_value=0.5, max_value=min(100.0, fs / 2 - 0.5 if fs > 1 else 50.0),
            value=min(20.0, max(1.0, fs / 4)) if fs > 0 else 20.0, step=1.0,
            key="jump_cutoff", disabled=not aplicar_filtro,
        )

    a_proc = a.copy()
    if aplicar_filtro and fs > 2 * cutoff_hz:
        a_proc = common.lowpass_filter(a, fs, cutoff_hz)
    elif aplicar_filtro:
        st.warning("Frequência de corte inválida para a taxa de amostragem estimada — filtro não aplicado.")

    # ---- detecção da fase de voo ----
    st.markdown("#### 🔍 Detecção da fase de voo")
    c1, c2 = st.columns(2)
    with c1:
        limiar = st.number_input(
            "Limiar de voo — |aceleração| abaixo disso (m/s²)",
            min_value=0.1, max_value=9.8, value=2.0, step=0.1, key="jump_limiar",
        )
    with c2:
        dur_min = st.number_input(
            "Duração mínima do voo (s)", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
            key="jump_dur_min", help="Use para ignorar quedas momentâneas de sinal por ruído.",
        )

    voo = _detectar_voo(t, a_proc, limiar, dur_min)
    if voo is None:
        st.error(
            "Não foi possível identificar uma fase de voo (nenhuma amostra abaixo do limiar). "
            "Ajuste o limiar acima ou confira o arquivo."
        )
        st.plotly_chart(
            common.plot_timeseries(t, {"Aceleração vertical": a_proc}, yaxis_title="m/s²"),
            use_container_width=True,
        )
        return

    s_voo, e_voo = voo
    resultados = _calcular_variaveis(t, a_proc, s_voo, e_voo)

    # ---- gráfico com marcações ----
    st.markdown("#### Sinal e pontos-chave")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a_proc, mode="lines+markers", name="Aceleração vertical",
                              line=dict(color="black", width=1.5), marker=dict(size=4)))
    fig.add_hline(y=limiar, line_dash="dot", line_color="#999", annotation_text="limiar de voo")
    fig.add_hline(y=-limiar, line_dash="dot", line_color="#999")

    marcadores = [
        ("Decolagem", resultados["t_decolagem"], "red"),
        ("Aterrissagem", resultados["t_aterrissagem"], "blue"),
    ]
    if resultados["t_pico_propulsao"] is not None:
        marcadores.append(("Pico propulsão", resultados["t_pico_propulsao"], "#2E7D32"))
    if resultados["t_contramovimento"] is not None:
        marcadores.append(("Contramovimento", resultados["t_contramovimento"], "#FBC02D"))
    if resultados["t_pico_pouso"] is not None:
        marcadores.append(("Pico pouso", resultados["t_pico_pouso"], "#8E24AA"))

    for label, tm, color in marcadores:
        fig.add_vline(x=tm, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top")

    fig.update_layout(
        xaxis_title="Tempo (s)", yaxis_title="Aceleração vertical (m/s²)",
        height=450, plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)

    # ---- relatório ----
    st.markdown("#### Resultados")
    linhas = {
        "Tempo de voo (s)": resultados["tempo_voo"],
        "Altura do salto — método do tempo de voo (cm)": resultados["altura_salto_cm"],
        "Velocidade de decolagem (m/s)": resultados["v_decolagem"],
        "Pico de aceleração — propulsão (m/s²)": resultados["pico_propulsao"],
        "Mínimo de aceleração — contramovimento (m/s²)": resultados["contramovimento_min"],
        "Pico de aceleração — aterrissagem (m/s²)": resultados["pico_pouso"],
    }
    df = pd.DataFrame(linhas.items(), columns=["Variável", "Valor"])
    df["Valor"] = df["Valor"].map(lambda v: f"{v:.3f}" if isinstance(v, (int, float)) and not np.isnan(v) else "—")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.info(
        f"**Altura estimada do salto: {resultados['altura_salto_cm']:.1f} cm** "
        f"(tempo de voo: {resultados['tempo_voo']*1000:.0f} ms)"
    )
    st.caption(
        "Altura = g·t²/8 (método do tempo de voo, g = 9,81 m/s²). Requer que o celular fique "
        "fixo e alinhado ao eixo vertical durante todo o salto para maior precisão."
    )
