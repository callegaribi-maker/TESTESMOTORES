"""
common.py — utilitários compartilhados pelos módulos do Momentum Web
(leitura de CSV/TXT do sensor inercial, filtros e plot padrão)
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import butter, filtfilt, detrend as sp_detrend


# -----------------------------
# Leitura de arquivo (Tempo, X, Y, Z)
# -----------------------------
def _decode_bytes(raw: bytes) -> str:
    """Decodifica bytes em texto tentando encodings comuns."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            text = raw.decode(enc)
            if text.strip():
                return text
        except Exception as e:
            last_err = e
    raise ValueError(f"Não foi possível decodificar o arquivo (encoding). Último erro: {last_err}")


def _infer_sep(text: str) -> str:
    """Inferência simples de separador."""
    sample = text[:8000]
    return ";" if sample.count(";") > sample.count(",") else ","


def to_float_series(s: pd.Series) -> pd.Series:
    """Converte série para float aceitando vírgula decimal."""
    s2 = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s2, errors="coerce")


def _drop_non_numeric_header_like_rows(df4: pd.DataFrame) -> pd.DataFrame:
    """Mantém apenas linhas onde Tempo,X,Y,Z conseguem virar número."""
    t = to_float_series(df4.iloc[:, 0])
    x = to_float_series(df4.iloc[:, 1])
    y = to_float_series(df4.iloc[:, 2])
    z = to_float_series(df4.iloc[:, 3])
    valid = t.notna() & x.notna() & y.notna() & z.notna()
    return df4.loc[valid].copy()


@st.cache_data(show_spinner=False)
def load_first4cols_cached(raw: bytes) -> pd.DataFrame:
    """
    Lê o arquivo sem cabeçalho e retorna dataframe [Tempo,X,Y,Z]
    usando as 4 primeiras colunas por posição. Cacheado.
    """
    text = _decode_bytes(raw)
    sep = _infer_sep(text)
    df = pd.read_csv(io.StringIO(text), sep=sep, header=None, engine="python")

    if df.shape[1] < 4:
        raise ValueError(f"O arquivo tem {df.shape[1]} colunas; preciso de pelo menos 4 (Tempo, X, Y, Z).")

    df4 = df.iloc[:, :4].copy()
    df4 = _drop_non_numeric_header_like_rows(df4)

    if df4.empty:
        raise ValueError("Após limpeza, não sobraram linhas numéricas válidas nas 4 primeiras colunas.")

    df4.columns = ["Tempo", "X", "Y", "Z"]
    return df4


def get_uploaded_series(uploader_label: str, state_key: str, help_text: str | None = None):
    """
    Renderiza um file_uploader, persiste os bytes em session_state (sobrevive a
    reruns) e retorna (t_sec, x, y, z) já limpos como arrays numpy, ou None.
    """
    uploaded = st.file_uploader(uploader_label, type=["csv", "txt"], key=f"{state_key}_uploader", help=help_text)

    raw_key = f"{state_key}_raw"
    if raw_key not in st.session_state:
        st.session_state[raw_key] = None
    if uploaded is not None:
        st.session_state[raw_key] = uploaded.getvalue()

    raw = st.session_state[raw_key]
    if not raw:
        st.info("Selecione um arquivo para iniciar.")
        return None

    try:
        df = load_first4cols_cached(raw)
    except Exception as e:
        st.error(f"Erro ao ler/processar o arquivo: {e}")
        return None

    t = to_float_series(df["Tempo"])
    x = to_float_series(df["X"])
    y = to_float_series(df["Y"])
    z = to_float_series(df["Z"])
    valid = t.notna() & x.notna() & y.notna() & z.notna()
    if valid.sum() < 5:
        st.error("Poucos pontos numéricos válidos após conversão. Verifique o arquivo.")
        return None

    t = t[valid].to_numpy(float)
    x = x[valid].to_numpy(float)
    y = y[valid].to_numpy(float)
    z = z[valid].to_numpy(float)

    # ordena por tempo e remove duplicados de timestamp
    order = np.argsort(t)
    t, x, y, z = t[order], x[order], y[order], z[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    t, x, y, z = t[keep], x[keep], y[keep], z[keep]

    t_sec = t / 1000.0 if t.max() > 1000 else t  # heurística: ms -> s
    return t_sec, x, y, z


# -----------------------------
# Processamento de sinal
# -----------------------------
def sampling_rate(t_sec: np.ndarray) -> float:
    dt = np.diff(t_sec)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.mean(dt))


def detrend(sig: np.ndarray) -> np.ndarray:
    return sp_detrend(sig)


def lowpass_filter(sig: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    wn = min(max(cutoff / nyq, 1e-4), 0.99)
    b, a = butter(order, wn, btype="low", analog=False)
    return filtfilt(b, a, sig)


def bandpass_filter(sig: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    lo = min(max(low / nyq, 1e-4), 0.98)
    hi = min(max(high / nyq, lo + 1e-3), 0.99)
    b, a = butter(order, [lo, hi], btype="band", analog=False)
    return filtfilt(b, a, sig)


# -----------------------------
# Plot padrão (mesmo visual do módulo Sensor Inercial)
# -----------------------------
def plot_timeseries(t_sec, series: dict, xaxis_title="Tempo (s)", yaxis_title="Amplitude",
                     legend_title="Sinais", markers=None):
    """
    series: {"nome": array}
    markers: lista opcional de dicts {"x": valor, "label": "texto", "color": "#hex"}
    """
    fig = go.Figure()
    for name, arr in series.items():
        fig.add_trace(go.Scatter(x=t_sec, y=arr, name=name, mode="lines"))

    if markers:
        for m in markers:
            fig.add_vline(x=m["x"], line_width=2, line_dash="dash",
                           line_color=m.get("color", "red"),
                           annotation_text=m.get("label", ""), annotation_position="top")

    fig.update_layout(
        xaxis=dict(title=dict(text=xaxis_title, font=dict(color="black", size=16)),
                   showline=True, linecolor="black", linewidth=2, mirror=True,
                   tickfont=dict(color="black", size=14), ticks="outside",
                   tickwidth=2, tickcolor="black"),
        yaxis=dict(title=dict(text=yaxis_title, font=dict(color="black", size=16)),
                   showline=True, linecolor="black", linewidth=2, mirror=True,
                   tickfont=dict(color="black", size=14), ticks="outside",
                   tickwidth=2, tickcolor="black"),
        legend_title=dict(text=legend_title, font=dict(color="black", size=14)),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig
