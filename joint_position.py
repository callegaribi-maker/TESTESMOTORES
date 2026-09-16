"""
joint_position.py — Módulo Posicionamento Articular (2 celulares, 2 segmentos)
(Momentum Web)

Adaptado do app de validação de flexão de cotovelo (Kinem x Celulares) que
você compartilhou — aqui não há um sistema de captura de movimento (Kinem)
como referência: o ângulo articular é estimado só a partir da INCLINAÇÃO
(acelerômetro) de dois celulares fixados em segmentos diferentes do membro
(ex.: braço/antebraço para cotovelo, coxa/perna para joelho). O giroscópio
de cada celular também é coletado, mas por enquanto só aparece como
referência visual/QA — o ângulo em si vem do acelerômetro, para evitar
deriva de integração.

Fluxo: upload dos 2-4 arquivos → classificação automática por nome →
corte automático de artefato no final → sincronização (automática pelo
1º pico de movimento de cada celular, com ajuste fino manual) → ângulo
articular contínuo → detecção de trials (picos) → calibração funcional
opcional → tabela de trials com ADM e erro vs. trial de referência →
métricas clássicas de posicionamento articular (AE/CE/VE).
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import find_peaks

import common

CATEGORIAS = [
    "Segmento 1 - Acelerômetro",
    "Segmento 1 - Giroscópio",
    "Segmento 2 - Acelerômetro",
    "Segmento 2 - Giroscópio",
    "Outro",
]


def _classificar_arquivo(nome_arquivo):
    n = nome_arquivo.lower()
    eh_seg1 = ("seg1" in n) or ("segmento1" in n) or ("segmento 1" in n) or ("s1" in n)
    eh_seg2 = ("seg2" in n) or ("segmento2" in n) or ("segmento 2" in n) or ("s2" in n)
    eh_acel = "acel" in n
    eh_gyro = ("gyro" in n) or ("giro" in n)
    if eh_seg1 and eh_acel:
        return CATEGORIAS[0]
    if eh_seg1 and eh_gyro:
        return CATEGORIAS[1]
    if eh_seg2 and eh_acel:
        return CATEGORIAS[2]
    if eh_seg2 and eh_gyro:
        return CATEGORIAS[3]
    return CATEGORIAS[4]


def _ler_arquivo(uploaded_file):
    raw = uploaded_file.getvalue()
    df = common.load_first4cols_cached(raw)  # colunas: Tempo, X, Y, Z
    t = common.to_float_series(df["Tempo"]).to_numpy(float)
    x = common.to_float_series(df["X"]).to_numpy(float)
    y = common.to_float_series(df["Y"]).to_numpy(float)
    z = common.to_float_series(df["Z"]).to_numpy(float)
    valid = ~np.isnan(t) & ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    t, x, y, z = t[valid], x[valid], y[valid], z[valid]
    order = np.argsort(t)
    return pd.DataFrame({"Tempo": t[order], "X": x[order], "Y": y[order], "Z": z[order]})


def _tempo_em_segundos(serie_tempo_ms):
    t = np.asarray(serie_tempo_ms, dtype=float)
    return t / 1000.0 if t.max() > 1000 else t


def _magnitude(df):
    return np.linalg.norm(df[["X", "Y", "Z"]].to_numpy(float), axis=1)


def _sugerir_corte(df, n_mad=8.0, fracao_cauda=0.05):
    """Corta artefato anômalo apenas na cauda final do registro (ex.: quando
    o celular é desligado/mexido no fim da captura)."""
    t = df["Tempo"].to_numpy(float)
    n = len(df)
    inicio_cauda = int(n * (1 - fracao_cauda))
    if inicio_cauda >= n:
        return float(t.max())
    scores = np.zeros(n)
    for c in ("X", "Y", "Z"):
        v = df[c].to_numpy(float)
        mediana = np.median(v)
        mad = np.median(np.abs(v - mediana)) * 1.4826
        if mad == 0:
            continue
        scores = np.maximum(scores, np.abs(v - mediana) / mad)
    scores_cauda = scores[inicio_cauda:]
    anomalos = np.where(scores_cauda > n_mad)[0]
    if len(anomalos) == 0:
        return float(t.max())
    idx_corte = max(0, inicio_cauda + anomalos[0] - 2)
    return float(t[idx_corte])


def _tilt_segmento(df):
    """Inclinação (graus) do eixo Y do celular em relação à vertical,
    a partir do acelerômetro (aproximação quase-estática)."""
    v = df[["X", "Y", "Z"]].to_numpy(float)
    mag = np.linalg.norm(v, axis=1)
    cos_a = v[:, 1] / mag
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def _detectar_pico_primeiro_movimento(tempo, valor, t_inicio=0.0, t_fim=None):
    tempo = np.asarray(tempo, dtype=float)
    valor = np.asarray(valor, dtype=float)
    if t_fim is None:
        t_fim = tempo.max()
    mask = (tempo >= t_inicio) & (tempo <= t_fim)
    if mask.sum() == 0:
        return None, None
    tw, vw = tempo[mask], valor[mask]
    baseline = np.median(vw)
    idx = int(np.argmax(np.abs(vw - baseline)))
    return float(tw[idx]), float(vw[idx])


def _angulo_articular(t1, tilt1, t2, tilt2):
    t0 = max(t1.min(), t2.min())
    t1_ = min(t1.max(), t2.max())
    if t1_ <= t0:
        return None, None
    grade = np.arange(t0, t1_, 0.02)
    i1 = np.interp(grade, t1, tilt1)
    i2 = np.interp(grade, t2, tilt2)
    return grade, np.abs(i1 - i2)


def _detectar_trials(tempo, sinal, prominence=15.0, distance_s=3.0, margem_extra_s=20.0):
    tempo = np.asarray(tempo, dtype=float)
    sinal = np.asarray(sinal, dtype=float)
    dt = np.median(np.diff(tempo))
    distance = max(1, int(distance_s / dt))
    picos, _ = find_peaks(sinal, prominence=prominence, distance=distance)
    if len(picos) == 0:
        return []
    limites = [0] + [int((picos[i] + picos[i + 1]) / 2) for i in range(len(picos) - 1)] + [len(sinal) - 1]
    margem_amostras = int(margem_extra_s / dt)
    trials = []
    for i, p in enumerate(picos):
        ini, fim = limites[i], limites[i + 1]
        segmento = sinal[ini:fim + 1]
        tempo_segmento = tempo[ini:fim + 1]
        ini_ext = max(0, p - margem_amostras)
        fim_ext = min(len(sinal) - 1, p + margem_amostras)
        trials.append({
            "trial": i + 1,
            "idx_pico": int(p),
            "tempo_pico": float(tempo[p]),
            "pico": float(sinal[p]),
            "adm": float(segmento.max() - segmento.min()),
            "tempo_rel": tempo_segmento - tempo[p],
            "sinal": segmento,
            "tempo_rel_ext": tempo[ini_ext:fim_ext + 1] - tempo[p],
            "sinal_ext": sinal[ini_ext:fim_ext + 1],
        })
    return trials


def _calcular_erros(trials, indice_referencia=0):
    if not trials:
        return trials
    pico_ref = trials[indice_referencia]["pico"]
    for t in trials:
        t["erro_abs"] = abs(t["pico"] - pico_ref)
        t["erro_rel_pct"] = (t["erro_abs"] / abs(pico_ref) * 100) if pico_ref != 0 else float("nan")
    return trials


def render():
    st.subheader("🦵 Posicionamento Articular — 2 celulares (acelerômetro + giroscópio)")
    st.caption(
        "Fixe um celular em cada segmento do membro avaliado (ex.: braço/antebraço para "
        "cotovelo, coxa/perna para joelho). O ângulo articular é estimado pela diferença "
        "de inclinação entre os dois celulares. Envie até 4 arquivos: acelerômetro e "
        "giroscópio de cada um dos dois segmentos."
    )

    c1, c2 = st.columns(2)
    nome_seg1 = c1.text_input("Nome do Segmento 1 (proximal)", value="Segmento 1", key="jps_nome1")
    nome_seg2 = c2.text_input("Nome do Segmento 2 (distal)", value="Segmento 2", key="jps_nome2")

    st.markdown("#### 📥 Arquivos")
    st.caption(
        "Dica: nomeie os arquivos com 'seg1'/'seg2' e 'acel'/'gyro' "
        "(ex.: seg1_acel.csv, seg1_gyro.csv, seg2_acel.csv, seg2_gyro.csv) "
        "para a classificação automática abaixo funcionar melhor."
    )
    arquivos = st.file_uploader(
        "Envie os arquivos juntos (Tempo, X, Y, Z)",
        type=["csv", "txt"], accept_multiple_files=True, key="jps_uploader",
    )
    if not arquivos:
        st.info("Envie os arquivos acima para começar.")
        return

    st.markdown("**Classificação detectada (ajuste se necessário)**")
    classificacoes = {}
    n_cols_class = min(len(arquivos), 4)
    cols_class = st.columns(n_cols_class)
    for i, arq in enumerate(arquivos):
        sugestao = _classificar_arquivo(arq.name)
        with cols_class[i % n_cols_class]:
            escolha = st.selectbox(arq.name, CATEGORIAS,
                                    index=CATEGORIAS.index(sugestao), key=f"jps_cat_{arq.name}")
        classificacoes[arq.name] = escolha

    dataframes = {}
    for arq in arquivos:
        cat = classificacoes[arq.name]
        try:
            df = _ler_arquivo(arq)
            chave = cat if cat not in dataframes else f"{cat} ({arq.name})"
            dataframes[chave] = df
        except Exception as e:
            st.error(f"Erro ao ler '{arq.name}': {e}")

    cat_acc1, cat_gyro1 = CATEGORIAS[0], CATEGORIAS[1]
    cat_acc2, cat_gyro2 = CATEGORIAS[2], CATEGORIAS[3]

    if cat_acc1 not in dataframes or cat_acc2 not in dataframes:
        st.warning(f"Preciso pelo menos do acelerômetro dos dois segmentos ('{cat_acc1}' e '{cat_acc2}').")
        return

    # ---- corte automático de artefato no final de cada arquivo ----
    for k in list(dataframes.keys()):
        df = dataframes[k]
        corte = _sugerir_corte(df)
        dataframes[k] = df[df["Tempo"] <= corte].reset_index(drop=True)

    tempo_seg = {k: _tempo_em_segundos(df["Tempo"].to_numpy(float)) for k, df in dataframes.items()}

    # ---- sincronização automática (1º pico de movimento em cada acelerômetro) ----
    JANELA = 15.0
    t1 = tempo_seg[cat_acc1]
    t2 = tempo_seg[cat_acc2]
    v1 = _magnitude(dataframes[cat_acc1])
    v2 = _magnitude(dataframes[cat_acc2])
    tp1, _ = _detectar_pico_primeiro_movimento(t1, v1, 0.0, JANELA)
    tp2, _ = _detectar_pico_primeiro_movimento(t2, v2, 0.0, JANELA)
    offset_auto = (tp1 - tp2) if (tp1 is not None and tp2 is not None) else 0.0

    with st.expander("🔧 Ajuste fino de sincronização (opcional)"):
        st.caption(
            f"Deslocamento aplicado a {nome_seg2} para alinhar com {nome_seg1}. "
            "Se as curvas não coincidirem no gráfico, ajuste manualmente até os "
            "picos de início de movimento coincidirem."
        )
        offset = st.number_input(f"Deslocamento — {nome_seg2} (s)",
                                  value=float(offset_auto), step=0.1, format="%.2f", key="jps_offset")

    t2_sync = t2 + offset

    tilt1 = _tilt_segmento(dataframes[cat_acc1])
    tilt2 = _tilt_segmento(dataframes[cat_acc2])

    grade, angulo = _angulo_articular(t1, tilt1, t2_sync, tilt2)
    if grade is None:
        st.error("Não há sobreposição de tempo suficiente entre os dois celulares após a sincronização.")
        return

    # ---- detecção de trials (tentativas) ----
    st.markdown("#### 🔍 Detecção de tentativas (trials)")
    pc1, pc2 = st.columns(2)
    prominence = pc1.number_input("Sensibilidade do pico (°)", value=15.0, step=1.0, min_value=1.0, key="jps_prom")
    distance_s = pc2.number_input("Distância mínima entre trials (s)", value=3.0, step=0.5, min_value=0.5, key="jps_dist")

    trials = _detectar_trials(grade, angulo, prominence, distance_s)
    if not trials:
        st.warning("Nenhum trial detectado — ajuste a sensibilidade acima.")
        fig0 = go.Figure(go.Scatter(x=grade, y=angulo, mode="lines", name="Ângulo"))
        fig0.update_layout(xaxis_title="Tempo (s)", yaxis_title="Ângulo articular (°)")
        st.plotly_chart(fig0, use_container_width=True)
        return

    # ---- calibração funcional (opcional) ----
    st.markdown("#### 🎯 Calibração funcional (opcional)")
    st.caption(
        "Se um dos trials corresponde a uma posição-alvo com ângulo conhecido "
        "(ex.: medido com goniômetro), informe aqui para corrigir a escala absoluta."
    )
    cal1, cal2, cal3 = st.columns(3)
    usar_cal = cal1.checkbox("Aplicar calibração", value=False, key="jps_usar_cal")
    offset_cal = 0.0
    if usar_cal:
        opcoes = [f"Trial {t['trial']}" for t in trials]
        trial_cal = cal2.selectbox("Trial de calibração", opcoes, key="jps_trial_cal")
        angulo_conhecido = cal3.number_input("Ângulo real conhecido (°)", value=90.0, step=1.0, key="jps_ang_conhecido")
        num_cal = int(trial_cal.replace("Trial ", ""))
        pico_bruto = next((t["pico"] for t in trials if t["trial"] == num_cal), None)
        if pico_bruto is not None:
            offset_cal = angulo_conhecido - pico_bruto
            st.success(f"Offset = {angulo_conhecido:.1f}° − {pico_bruto:.1f}° (bruto) = **{offset_cal:+.1f}°**")

    if offset_cal != 0.0:
        angulo = angulo + offset_cal
        for t in trials:
            t["pico"] += offset_cal
            t["sinal"] = t["sinal"] + offset_cal
            t["sinal_ext"] = t["sinal_ext"] + offset_cal

    # ---- gráfico contínuo com trials marcados ----
    st.markdown("#### Sinal contínuo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grade, y=angulo, mode="lines", name=f"Ângulo ({nome_seg1} × {nome_seg2})"))
    fig.add_trace(go.Scatter(
        x=[t["tempo_pico"] for t in trials], y=[t["pico"] for t in trials],
        mode="markers+text", text=[f"T{t['trial']}" for t in trials],
        textposition="top center", marker=dict(size=9, symbol="diamond", color="crimson"),
        name="Trials",
    ))
    fig.update_layout(xaxis_title="Tempo (s)", yaxis_title="Ângulo articular (°)", height=420)
    st.plotly_chart(fig, use_container_width=True)

    # ---- seleção de trials incluídos + trial de referência ----
    st.markdown("#### Trials incluídos e referência")
    todos = [t["trial"] for t in trials]
    incluidos = st.multiselect("Trials incluídos", options=todos, default=todos, key="jps_incluidos")
    trials_f = [t for t in trials if t["trial"] in incluidos]
    if not trials_f:
        st.info("Nenhum trial incluído.")
        return
    opcoes_ref = [f"Trial {t['trial']}" for t in trials_f]
    ref_escolhido = st.selectbox(
        "Trial de referência (ex.: posição-alvo memorizada)", opcoes_ref, index=0, key="jps_ref"
    )
    idx_ref = opcoes_ref.index(ref_escolhido)
    trials_f = _calcular_erros(trials_f, idx_ref)

    # ---- tabela de trials ----
    st.markdown("#### Tabela — trials, ADM e erro vs. referência")
    df_tab = pd.DataFrame(trials_f)[["trial", "tempo_pico", "pico", "adm", "erro_abs", "erro_rel_pct"]]
    df_tab.columns = ["Trial", "t pico (s)", "Pico (°)", "ADM (°)", "Erro abs (°)", "Erro rel (%)"]
    resumo = pd.DataFrame({
        "Trial": ["Média", "Desvio padrão (SD)"],
        "t pico (s)": [df_tab["t pico (s)"].mean(), df_tab["t pico (s)"].std()],
        "Pico (°)": [df_tab["Pico (°)"].mean(), df_tab["Pico (°)"].std()],
        "ADM (°)": [df_tab["ADM (°)"].mean(), df_tab["ADM (°)"].std()],
        "Erro abs (°)": [df_tab["Erro abs (°)"].mean(), df_tab["Erro abs (°)"].std()],
        "Erro rel (%)": [df_tab["Erro rel (%)"].mean(), df_tab["Erro rel (%)"].std()],
    })
    st.dataframe(pd.concat([df_tab, resumo], ignore_index=True).round(2), use_container_width=True, hide_index=True)

    # ---- métricas clássicas de posicionamento articular (AE/CE/VE) ----
    st.markdown("#### Resumo — Métricas clássicas de JPS")
    pico_ref = trials_f[idx_ref]["pico"]
    erros_assinados = [t["pico"] - pico_ref for t in trials_f if t["trial"] != trials_f[idx_ref]["trial"]]
    if erros_assinados:
        ae = float(np.mean(np.abs(erros_assinados)))
        ce = float(np.mean(erros_assinados))
        ve = float(np.std(erros_assinados, ddof=1)) if len(erros_assinados) > 1 else np.nan
        r1, r2, r3 = st.columns(3)
        r1.metric("Erro Absoluto médio (AE)", f"{ae:.2f}°")
        r2.metric("Erro Constante (CE) — viés", f"{ce:+.2f}°")
        r3.metric("Erro Variável (VE) — SD", f"{ve:.2f}°" if not np.isnan(ve) else "—")
        st.caption(
            "AE: precisão geral (quanto menor, melhor). CE: tendência a superestimar (+) ou "
            "subestimar (−) o ângulo-alvo. VE: consistência entre as tentativas de reprodução."
        )
    else:
        st.info("Inclua pelo menos 2 trials (1 de referência + 1 de reprodução) para calcular AE/CE/VE.")

    # ---- gráfico de barras por trial ----
    fig_bar = go.Figure(go.Bar(
        x=[f"Trial {t['trial']}" for t in trials_f], y=[t["pico"] for t in trials_f],
        marker_color="#1f77b4", text=[f"{t['pico']:.1f}°" for t in trials_f], textposition="outside",
    ))
    fig_bar.update_layout(title="Ângulo de pico por trial", xaxis_title="Trial", yaxis_title="Ângulo (°)", height=380)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ---- giroscópio (referência visual, não usado no cálculo do ângulo) ----
    with st.expander("📡 Giroscópio (referência visual — ainda não usado no cálculo do ângulo)"):
        st.caption(
            "O ângulo é calculado só a partir do acelerômetro (evita deriva de integração). "
            "O giroscópio abaixo é apenas para conferência visual do movimento."
        )
        if cat_gyro1 in dataframes:
            tg1 = _tempo_em_segundos(dataframes[cat_gyro1]["Tempo"].to_numpy(float))
            mag_g1 = _magnitude(dataframes[cat_gyro1])
            fig_g1 = go.Figure(go.Scatter(x=tg1, y=mag_g1, name=f"Giro {nome_seg1}"))
            fig_g1.update_layout(xaxis_title="Tempo (s)", yaxis_title="Vel. angular (mag.)", height=280)
            st.plotly_chart(fig_g1, use_container_width=True)
        if cat_gyro2 in dataframes:
            tg2 = _tempo_em_segundos(dataframes[cat_gyro2]["Tempo"].to_numpy(float)) + offset
            mag_g2 = _magnitude(dataframes[cat_gyro2])
            fig_g2 = go.Figure(go.Scatter(x=tg2, y=mag_g2, name=f"Giro {nome_seg2}"))
            fig_g2.update_layout(xaxis_title="Tempo (s)", yaxis_title="Vel. angular (mag.)", height=280)
            st.plotly_chart(fig_g2, use_container_width=True)
