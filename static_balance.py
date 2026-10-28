"""
static_balance.py — Módulo de Equilíbrio Estático (Momentum Web)

Protocolo esperado: celular fixado próximo ao centro de massa (cintura/tronco
baixo), paciente em pé e parado. Arquivo CSV/TXT: Tempo, X, Y, Z
(acelerômetro, em m/s² ou g).

Processamento alinhado ao racional de Almeida, Callegari, Souza et al.
(LEMOH/UFPA) — "Smartphone accelerometry for postural balance assessment":
  - eixos AP/ML explicitados (no artigo: Z do dispositivo = AP, X = ML,
    para celular fixado em L5) em vez de usar X/Y genéricos;
  - sinal centrado na média e com tendência linear removida (detrend linear);
  - filtro passa-baixa Butterworth de fase zero, por padrão 2ª ordem a 6 Hz
    (valores fixos do artigo — ainda editáveis em modo avançado);
  - janela de análise opcional de duração fixa com início a partir de um
    offset configurável (padrão: 45 s a partir de 10 s), para descartar o
    transiente inicial de acomodação na posição;
  - variáveis globais: RMS-AP, RMS-ML, path length ("aceleração total"),
    área da elipse de confiança 95% e seus dois semieixos (a, b) — mesmo
    conjunto do artigo (RMS, TD, área/semieixos da elipse). SDA (Stabilogram
    Diffusion Analysis) não é calculada aqui: o próprio artigo conclui que
    ela não acrescenta poder discriminativo além dessas variáveis globais.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import common


def _sway_metrics(ap: np.ndarray, ml: np.ndarray, t_sec: np.ndarray) -> dict:
    """
    ap, ml: sinais já centrados/detrendados/filtrados correspondentes aos
    eixos anteroposterior (AP) e mediolateral (ML).
    """
    duration = t_sec[-1] - t_sec[0] if len(t_sec) > 1 else np.nan

    # Path length ("aceleração total") — Σ√(ΔAP² + ΔML²)
    path = np.sum(np.sqrt(np.diff(ap) ** 2 + np.diff(ml) ** 2))
    mean_velocity = path / duration if duration else np.nan

    rms_ap = float(np.sqrt(np.mean(ap ** 2)))
    rms_ml = float(np.sqrt(np.mean(ml ** 2)))
    range_ap = float(np.ptp(ap))
    range_ml = float(np.ptp(ml))

    # Elipse de confiança 95% (autovalores da matriz de covariância AP-ML)
    cov = np.cov(ap, ml)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0, None)
    chi2_95 = 5.991  # qui-quadrado, 2 graus de liberdade, 95%
    semi_a = float(np.sqrt(chi2_95 * eigvals[1]))  # maior semieixo
    semi_b = float(np.sqrt(chi2_95 * eigvals[0]))  # menor semieixo
    ellipse_area = float(np.pi * semi_a * semi_b)

    return {
        "Duração analisada (s)": duration,
        "RMS AP (m/s²)": rms_ap,
        "RMS ML (m/s²)": rms_ml,
        "Amplitude AP (m/s²)": range_ap,
        "Amplitude ML (m/s²)": range_ml,
        "Comprimento do sway / aceleração total (m/s²)": float(path),
        "Velocidade média de sway (m/s²·s⁻¹)": float(mean_velocity),
        "Semieixo maior da elipse — a (m/s²)": semi_a,
        "Semieixo menor da elipse — b (m/s²)": semi_b,
        "Área da elipse 95% (m/s²)²": ellipse_area,
    }


def render():
    st.subheader("⚖️ Equilíbrio Estático")
    st.caption(
        "Celular fixado próximo ao centro de massa (cintura/L5), paciente em pé, parado. "
        "Envie o arquivo CSV/TXT exportado do sensor inercial (Tempo, X, Y, Z)."
    )

    condicao = st.selectbox("Condição do teste", ["Olhos abertos", "Olhos fechados", "Outra"])

    data = common.get_uploaded_series("Selecione o arquivo CSV/TXT", "static_balance")
    if data is None:
        return

    t_sec, x, y, z = data
    fs = common.sampling_rate(t_sec)

    # ---------------------------------------------------------------
    # Mapeamento de eixos AP/ML
    # ---------------------------------------------------------------
    st.markdown("#### 📍 Mapeamento dos eixos (AP / ML)")
    st.caption(
        "No protocolo de referência (celular fixado em L5), o eixo Z do aparelho "
        "corresponde ao sentido anteroposterior (AP) e o eixo X ao mediolateral (ML). "
        "Ajuste abaixo se a fixação/orientação do seu celular for diferente."
    )
    eixo_opcoes = {"X": x, "Y": y, "Z": z}
    col_ap, col_ml = st.columns(2)
    with col_ap:
        eixo_ap_nome = st.selectbox("Eixo = Anteroposterior (AP)", list(eixo_opcoes.keys()),
                                     index=2, key="sb_eixo_ap")  # default Z
    with col_ml:
        opcoes_ml = [k for k in eixo_opcoes.keys() if k != eixo_ap_nome] + [eixo_ap_nome]
        default_ml = "X" if eixo_ap_nome != "X" else "Y"
        eixo_ml_nome = st.selectbox("Eixo = Mediolateral (ML)", list(eixo_opcoes.keys()),
                                     index=list(eixo_opcoes.keys()).index(default_ml), key="sb_eixo_ml")

    ap_raw = eixo_opcoes[eixo_ap_nome].copy()
    ml_raw = eixo_opcoes[eixo_ml_nome].copy()

    # ---------------------------------------------------------------
    # Janela de análise (padrão do artigo: 45 s a partir de 10 s)
    # ---------------------------------------------------------------
    st.markdown("#### ⏱️ Janela de análise")
    usar_janela = st.checkbox(
        "Aplicar janela de análise fixa (descarta o início — transiente de acomodação)",
        value=True, key="sb_usar_janela",
    )
    total_dur = float(t_sec[-1] - t_sec[0]) if len(t_sec) > 1 else 0.0
    t0_rel = 0.0
    if usar_janela:
        c_ini, c_dur = st.columns(2)
        with c_ini:
            inicio_s = st.number_input(
                "Início da janela — offset após o começo do registro (s)",
                min_value=0.0, max_value=max(0.0, total_dur), value=min(10.0, max(0.0, total_dur)),
                step=1.0, key="sb_janela_inicio",
            )
        with c_dur:
            duracao_s = st.number_input(
                "Duração da janela (s)", min_value=1.0, max_value=max(1.0, total_dur),
                value=float(min(45.0, max(1.0, total_dur))), step=1.0, key="sb_janela_duracao",
            )
        t0_rel = t_sec[0] + inicio_s
        t1_rel = t0_rel + duracao_s
        mask = (t_sec >= t0_rel) & (t_sec <= t1_rel)
        if mask.sum() < 5:
            st.warning(
                "A janela escolhida (início + duração) é maior do que o registro disponível "
                f"({total_dur:.1f} s). Usando o sinal completo."
            )
        else:
            t_sec, ap_raw, ml_raw = t_sec[mask], ap_raw[mask], ml_raw[mask]
            if mask.sum() < len(mask):
                st.caption(
                    f"Janela aplicada: {inicio_s:.0f}–{inicio_s + duracao_s:.0f} s "
                    f"({mask.sum()} amostras)."
                )

    # ---------------------------------------------------------------
    # Pré-processamento: detrend + filtro passa-baixa
    # ---------------------------------------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        apply_detrend = st.checkbox(
            "Aplicar detrend (centrar na média + remover tendência linear)", True,
        )
    with col_b:
        apply_filter = st.checkbox("Aplicar filtro passa-baixa", True)

    ap_f, ml_f = ap_raw.copy(), ml_raw.copy()
    if apply_detrend:
        ap_f, ml_f = common.detrend(ap_f), common.detrend(ml_f)

    if apply_filter and fs > 0:
        with st.expander("⚙️ Parâmetros do filtro (padrão = protocolo de referência: Butterworth 2ª ordem, 6 Hz)"):
            modo_avancado = st.checkbox(
                "Ajustar manualmente ordem/corte (modo avançado)", value=False, key="sb_filtro_avancado",
            )
            if modo_avancado:
                ordem = st.number_input("Ordem do filtro", min_value=1, max_value=8, value=2, step=1, key="sb_ordem")
                cutoff = st.slider(
                    "Frequência de corte (Hz)", 0.5, min(20.0, fs / 2 - 0.5), 6.0, 0.5, key="sb_cutoff",
                )
            else:
                ordem, cutoff = 2, 6.0
                st.caption(f"Usando ordem={ordem}, corte={cutoff} Hz (fixo, igual ao artigo de referência).")
        cutoff = min(cutoff, fs / 2 - 0.5) if fs > 0 else cutoff
        if cutoff > 0 and fs / 2 > cutoff:
            ap_f = common.lowpass_filter(ap_f, fs, cutoff, order=int(ordem))
            ml_f = common.lowpass_filter(ml_f, fs, cutoff, order=int(ordem))
        else:
            st.warning("Taxa de amostragem insuficiente para a frequência de corte escolhida — filtro não aplicado.")

    st.plotly_chart(
        common.plot_timeseries(
            t_sec, {f"AP ({eixo_ap_nome})": ap_f, f"ML ({eixo_ml_nome})": ml_f},
            yaxis_title="Aceleração (m/s²)",
        ),
        use_container_width=True,
    )

    st.markdown("#### Estatocinesiograma (AP vs ML)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ml_f, y=ap_f, mode="lines+markers",
                              marker=dict(size=3), line=dict(width=1)))
    fig.update_layout(
        xaxis=dict(title="ML (m/s²)", showline=True, linecolor="black", mirror=True, scaleanchor="y"),
        yaxis=dict(title="AP (m/s²)", showline=True, linecolor="black", mirror=True),
    )
    st.plotly_chart(fig, use_container_width=True)

    metrics = _sway_metrics(ap_f, ml_f, t_sec)
    st.markdown(f"#### Resultados — {condicao}")
    df_metrics = pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"])
    df_metrics["Valor"] = df_metrics["Valor"].map(lambda v: f"{v:.4f}")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    st.caption(
        "Métricas calculadas a partir da aceleração (não do centro de pressão), seguindo o "
        "racional de Almeida, Callegari, Souza et al. (LEMOH/UFPA): detrend linear, filtro "
        "passa-baixa Butterworth de fase zero (2ª ordem, 6 Hz por padrão), RMS/path length/"
        "elipse de confiança 95% calculados sobre os eixos AP e ML mapeados acima. Use-as de "
        "forma comparativa (ex.: olhos abertos vs. fechados, pré vs. pós). A análise de difusão "
        "do estabilograma (SDA) não é calculada neste módulo."
    )
