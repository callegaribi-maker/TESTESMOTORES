"""
sit_to_stand.py — Módulo Sentar e Levantar (Sit-to-Stand) (Momentum Web)

Ferramenta de análise de sinal por picos/ciclos: upload de arquivo (CSV/TXT/
Excel) em formato livre (você escolhe as colunas de tempo e sinal), detecção
automática de picos (cada pico = transição sentar/levantar), edição manual
dos picos (adicionar/remover), tabela de ciclos, sobreposição normalizada
(0→1) com curva média ± DP, e acúmulo de resultados de vários participantes
com exportação em Excel.
"""
import io
from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.signal import find_peaks


def render():
    st.subheader("🪑 Sentar e Levantar (Sit-to-Stand) — Análise de Picos e Ciclos")
    st.caption(
        "Celular no tronco (peito ou região lombar) durante as repetições. "
        "Cada pico detectado no sinal representa uma transição sentar/levantar."
    )

    # ── Session state (namespace 'sts_' para não colidir com outros módulos) ──
    if "sts_peaks" not in st.session_state:
        st.session_state.sts_peaks = []
    if "sts_trigger_auto" not in st.session_state:
        st.session_state.sts_trigger_auto = False
    if "sts_saved_results" not in st.session_state:
        st.session_state.sts_saved_results = []  # [{name, resultante_df, mean_dur, std_dur, n_cycles}]

    # ── Upload e configurações (página normal, sem barra lateral) ─────────────
    st.markdown("#### 📥 Arquivo de sinal")
    uploaded = st.file_uploader(
        "Selecione o arquivo (CSV, TXT ou Excel)",
        type=["csv", "txt", "xlsx", "xls"], key="sts_uploader"
    )

    df = None
    x_col = y_col = None

    if uploaded:
        ext = uploaded.name.rsplit(".", 1)[-1].lower()
        if ext in ("xlsx", "xls"):
            try:
                df = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f"Erro ao ler Excel: {e}")
        else:
            c_sep, c_header, c_dec = st.columns(3)
            with c_sep:
                sep = st.selectbox(
                    "Separador",
                    [";", ",", "\t", " "],
                    format_func=lambda s: {
                        ";": "Ponto-e-vírgula (;)",
                        ",": "Vírgula (,)",
                        "\t": "Tab",
                        " ": "Espaço",
                    }[s],
                    index=0,
                    key="sts_sep",
                )
            with c_header:
                header_row = st.number_input(
                    "Linha do cabeçalho (0 = primeira)",
                    min_value=0, max_value=20, value=0, key="sts_header_row",
                )
            with c_dec:
                decimal = st.selectbox(
                    "Decimal",
                    [".", ","],
                    format_func=lambda s: {
                        ".": "Ponto . (padrão EN)",
                        ",": "Vírgula , (padrão BR)",
                    }[s],
                    index=0,
                    key="sts_decimal",
                )
            try:
                raw = uploaded.read()
                try:
                    text = raw.decode("utf-8-sig")
                except Exception:
                    text = raw.decode("latin-1")
                df = pd.read_csv(
                    io.StringIO(text),
                    sep=sep,
                    header=int(header_row),
                    decimal=decimal,
                    engine="python",
                )
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")

        if df is not None:
            cols = df.columns.tolist()
            default_x = "DURACAO" if "DURACAO" in cols else cols[0]
            default_y = "ACC EIXO Y" if "ACC EIXO Y" in cols else cols[-1]
            c_x, c_y = st.columns(2)
            with c_x:
                x_col = st.selectbox(
                    "Coluna X (tempo)",
                    cols,
                    index=cols.index(default_x) if default_x in cols else 0,
                    key="sts_xcol",
                )
            with c_y:
                y_col = st.selectbox(
                    "Coluna Y (sinal)",
                    cols,
                    index=cols.index(default_y) if default_y in cols else 0,
                    key="sts_ycol",
                )

    st.divider()
    st.markdown("#### 🔍 Detecção automática de picos")
    c_prom, c_dist = st.columns(2)
    with c_prom:
        prominence_pct = st.slider("Proeminência mínima (%)", 1, 80, 20, key="sts_prom")
    with c_dist:
        min_dist_pct = st.slider("Distância mínima entre picos (%)", 1, 30, 5, key="sts_dist")
    if st.button("Auto-detectar picos", use_container_width=True, key="sts_autodetect"):
        st.session_state.sts_trigger_auto = True

    with st.expander("⚙️ Configuração avançada"):
        snap_pct = st.slider(
            "Snap para máximo local (%)", 0, 10, 2, key="sts_snap",
            help="Janela ao redor do X digitado para encaixar no máximo local",
        )

    # ── Sem arquivo ──────────────────────────────────────────────────────────
    if df is None:
        st.info("👆 Carregue um arquivo acima para começar.")
        with st.expander("📖 Como usar"):
            st.markdown(
                """
                1. **Carregue** um arquivo CSV, TXT ou Excel na barra lateral
                2. Verifique as colunas X e Y selecionadas
                3. Use **Auto-detectar** para encontrar picos automaticamente
                4. **Clique no triângulo** para remover um pico
                5. Use o campo abaixo do gráfico para **adicionar** um pico manualmente
                6. Baixe ciclos, picos e matriz em Excel
                """
            )
        return

    # ── Preparar dados ───────────────────────────────────────────────────────
    x = pd.to_numeric(df[x_col], errors="coerce").values.astype(float)
    y = pd.to_numeric(df[y_col], errors="coerce").values.astype(float)
    valid = ~(np.isnan(x) | np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)

    if n == 0:
        st.error("Nenhum dado numérico válido. Verifique separador e decimal.")
        return

    # ── Auto-detecção ────────────────────────────────────────────────────────
    if st.session_state.sts_trigger_auto:
        y_range = np.ptp(y) or 1.0
        prom = prominence_pct / 100 * y_range
        dist = max(1, int(n * min_dist_pct / 100))
        idx_peaks, _ = find_peaks(y, prominence=prom, distance=dist)
        st.session_state.sts_peaks = [
            {"x": float(x[i]), "y": float(y[i])} for i in idx_peaks
        ]
        st.session_state.sts_trigger_auto = False

    sorted_peaks = sorted(st.session_state.sts_peaks, key=lambda p: p["x"])

    # ── Figura principal ─────────────────────────────────────────────────────
    fig = go.Figure()

    cycle_colors = ["rgba(99,200,99,0.12)", "rgba(99,149,255,0.12)"]
    for i in range(len(sorted_peaks) - 1):
        fig.add_vrect(
            x0=sorted_peaks[i]["x"], x1=sorted_peaks[i + 1]["x"],
            fillcolor=cycle_colors[i % 2],
            layer="below", line_width=0,
            annotation_text=f" C{i + 1}",
            annotation_position="top left",
            annotation=dict(font_size=10, font_color="#777"),
        )

    for p in sorted_peaks:
        fig.add_vline(
            x=p["x"],
            line=dict(color="rgba(200,50,50,0.3)", width=1, dash="dot"),
        )

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines", name="Sinal",
        line=dict(color="#4C78A8", width=1.5),
        hovertemplate="x: %{x:.4g}<br>y: %{y:.4g}<extra>Sinal</extra>",
    ))

    if sorted_peaks:
        fig.add_trace(go.Scatter(
            x=[p["x"] for p in sorted_peaks],
            y=[p["y"] for p in sorted_peaks],
            mode="markers+text",
            name="Picos",
            text=[f"P{i + 1}" for i in range(len(sorted_peaks))],
            textposition="top center",
            textfont=dict(size=10, color="darkred"),
            marker=dict(color="crimson", size=11, symbol="triangle-up",
                        line=dict(color="darkred", width=1)),
            hovertemplate="Pico x=%{x:.4g}<br>y=%{y:.4g}<extra>Pico</extra>",
        ))

    fig.update_layout(
        height=500,
        title="<b>Clique no triângulo</b> para remover pico",
        xaxis_title="Tempo (ms)",
        yaxis_title=y_col,
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=70, b=10),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=False)

    # ── Remoção por clique no triângulo ─────────────────────────────────────
    selected = st.plotly_chart(
        fig, on_select="rerun", selection_mode="points",
        key="sts_signal_plot", use_container_width=True,
    )

    if selected and selected.selection and selected.selection.points:
        pt = selected.selection.points[0]
        if sorted_peaks and pt.get("curve_number", 0) == 1:
            pt_idx = pt.get("point_index", 0)
            if 0 <= pt_idx < len(sorted_peaks):
                target_x = sorted_peaks[pt_idx]["x"]
                x_tol = np.ptp(x) * 0.01
                st.session_state.sts_peaks = [
                    p for p in st.session_state.sts_peaks
                    if abs(p["x"] - target_x) > x_tol * 0.1
                ]
                st.rerun()

    # ── Adicionar pico manualmente ───────────────────────────────────────────
    st.subheader("➕ Adicionar pico")
    st.caption("Use o zoom do gráfico para encontrar o X desejado, depois digite aqui.")
    col_add, col_clear = st.columns([2, 1])

    with col_add:
        add_x = st.number_input(
            "Valor X do pico",
            min_value=float(x.min()), max_value=float(x.max()),
            value=float(x[n // 2]),
            step=float(np.ptp(x) / n * 10),
            format="%.1f",
            key="sts_add_x",
        )
        if st.button("➕ Adicionar pico nesse X", use_container_width=True, key="sts_add_btn"):
            idx_click = int(np.argmin(np.abs(x - add_x)))
            half_w = max(1, int(n * snap_pct / 100))
            i0 = max(0, idx_click - half_w)
            i1 = min(n - 1, idx_click + half_w)
            idx_max = i0 + int(np.argmax(y[i0: i1 + 1]))
            new_x, new_y = float(x[idx_max]), float(y[idx_max])
            x_tol = np.ptp(x) * 0.01
            if not any(abs(p["x"] - new_x) < x_tol for p in st.session_state.sts_peaks):
                st.session_state.sts_peaks.append({"x": new_x, "y": new_y})
                st.rerun()
            else:
                st.warning("Já existe um pico próximo desse X.")

    with col_clear:
        st.write("")
        st.write("")
        if st.button("🗑️ Limpar todos", use_container_width=True, key="sts_clear_btn"):
            st.session_state.sts_peaks = []
            st.rerun()

    st.caption(f"**{len(sorted_peaks)} pico(s)** · Clique no triângulo vermelho para remover")

    # ── Tabela de ciclos ─────────────────────────────────────────────────────
    if len(sorted_peaks) >= 2:
        st.subheader(f"📊 {len(sorted_peaks) - 1} ciclo(s) detectado(s)")

        rows = []
        for i in range(len(sorted_peaks) - 1):
            xs, xe = sorted_peaks[i]["x"], sorted_peaks[i + 1]["x"]
            mask = (x >= xs) & (x <= xe)
            cy = y[mask]
            rows.append({
                "Ciclo": i + 1,
                "X início": round(xs, 3),
                "X fim": round(xe, 3),
                "Duração (ms)": round(xe - xs, 3),
                "Pontos": int(mask.sum()),
                "Máx.": round(float(cy.max()), 5) if cy.size else "—",
                "Mín.": round(float(cy.min()), 5) if cy.size else "—",
                "Média": round(float(cy.mean()), 5) if cy.size else "—",
                "RMS": round(float(np.sqrt(np.mean(cy ** 2))), 5) if cy.size else "—",
            })

        cycles_df = pd.DataFrame(rows)
        st.dataframe(cycles_df, use_container_width=True, hide_index=True)

        peaks_df = pd.DataFrame(
            [{"Pico": i + 1, "x": p["x"], "y": p["y"]}
             for i, p in enumerate(sorted_peaks)]
        )
        buf_info = BytesIO()
        with pd.ExcelWriter(buf_info, engine="openpyxl") as writer:
            cycles_df.to_excel(writer, sheet_name="Ciclos", index=False)
            peaks_df.to_excel(writer, sheet_name="Picos", index=False)
        st.download_button(
            "⬇️ Baixar ciclos e picos (Excel)",
            buf_info.getvalue(),
            "ciclos_picos.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="sts_dl_ciclos",
        )

        # ── Segmentos de ciclo ───────────────────────────────────────────────
        cycles_seg = []
        for i in range(len(sorted_peaks) - 1):
            xs, xe = sorted_peaks[i]["x"], sorted_peaks[i + 1]["x"]
            mask = (x >= xs) & (x <= xe)
            cx, cy2 = x[mask], y[mask]
            if len(cx) >= 2:
                cycles_seg.append({
                    "label": f"C{i + 1}",
                    "x": cx, "y": cy2,
                    "duration": xe - xs,
                })

        if len(cycles_seg) >= 2:
            st.divider()
            st.header("📈 Análise de Ciclos")

            PALETTE = [
                "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
                "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
                "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
            ]

            tab1, tab2 = st.tabs(["⏱️ Duração original", "📐 Duração normalizada (0 → 1)"])

            # Tab 1 — duração original
            with tab1:
                fig_orig = go.Figure()
                for i, cyc in enumerate(cycles_seg):
                    x_rel = cyc["x"] - cyc["x"][0]
                    fig_orig.add_trace(go.Scatter(
                        x=x_rel, y=cyc["y"],
                        mode="lines", name=cyc["label"],
                        line=dict(color=PALETTE[i % len(PALETTE)], width=1.2),
                        opacity=0.75,
                    ))
                fig_orig.update_layout(
                    width=650, height=650,
                    title="Ciclos sobrepostos — tempo relativo ao início de cada ciclo",
                    xaxis_title="Tempo (ms)",
                    yaxis_title=y_col,
                    hovermode="x unified",
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", y=-0.2),
                    margin=dict(l=10, r=10, t=50, b=80),
                )
                fig_orig.update_xaxes(showgrid=True, gridcolor="#eee")
                fig_orig.update_yaxes(showgrid=True, gridcolor="#eee")
                st.plotly_chart(fig_orig, use_container_width=False, key="sts_fig_orig")

            # Tab 2 — normalizado
            with tab2:
                N_NORM = 300
                x_norm = np.linspace(0, 1, N_NORM)
                all_y_norm = []

                durations_pre = [cyc["duration"] for cyc in cycles_seg]
                mean_dur_pre = float(np.mean(durations_pre))

                fig_norm = go.Figure()
                for i, cyc in enumerate(cycles_seg):
                    x_rel = (cyc["x"] - cyc["x"][0]) / (cyc["x"][-1] - cyc["x"][0])
                    y_interp = np.interp(x_norm, x_rel, cyc["y"])
                    all_y_norm.append(y_interp)
                    fig_norm.add_trace(go.Scatter(
                        x=x_norm, y=y_interp,
                        mode="lines", name=cyc["label"],
                        line=dict(color=PALETTE[i % len(PALETTE)], width=1),
                        opacity=0.45,
                    ))

                mean_y = np.mean(all_y_norm, axis=0)
                std_y = np.std(all_y_norm, axis=0)

                fig_norm.add_trace(go.Scatter(
                    x=np.concatenate([x_norm, x_norm[::-1]]),
                    y=np.concatenate([mean_y + std_y, (mean_y - std_y)[::-1]]),
                    fill="toself", fillcolor="rgba(0,0,0,0.08)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="±1 DP", showlegend=True,
                ))
                fig_norm.add_trace(go.Scatter(
                    x=x_norm, y=mean_y,
                    mode="lines", name="Média",
                    line=dict(color="black", width=2.5),
                ))

                fig_norm.update_layout(
                    width=650, height=650,
                    title="Ciclos normalizados (0 → 1) com curva média ± 1 DP",
                    xaxis_title="Fase normalizada",
                    yaxis_title=y_col,
                    hovermode="x unified",
                    plot_bgcolor="white", paper_bgcolor="white",
                    legend=dict(orientation="h", y=-0.2),
                    margin=dict(l=10, r=10, t=50, b=80),
                )
                fig_norm.update_xaxes(showgrid=True, gridcolor="#eee")
                fig_norm.update_yaxes(showgrid=True, gridcolor="#eee")
                st.plotly_chart(fig_norm, use_container_width=False, key="sts_fig_norm")

            # ── Estatísticas ──────────────────────────────────────────────────
            durations = durations_pre
            mean_dur = mean_dur_pre
            std_dur = float(np.std(durations))
            cv_dur = std_dur / mean_dur * 100 if mean_dur else 0

            st.subheader("⏱️ Estatísticas de duração")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Nº de ciclos", len(cycles_seg))
            m2.metric("Duração média", f"{mean_dur:.1f} ms")
            m3.metric("Desvio padrão", f"{std_dur:.1f} ms")
            m4.metric("CV (%)", f"{cv_dur:.1f}")

            # ── Exportar ─────────────────────────────────────────────────────
            st.subheader("📤 Exportar matrizes")

            matrix = {"fase_norm": np.round(x_norm, 5)}
            for i, (cyc, y_interp) in enumerate(zip(cycles_seg, all_y_norm)):
                matrix[cyc["label"]] = np.round(y_interp, 6)
            matrix["Media"] = np.round(mean_y, 6)
            matrix["DP"] = np.round(std_y, 6)
            matrix["Media_+DP"] = np.round(mean_y + std_y, 6)
            matrix["Media_-DP"] = np.round(mean_y - std_y, 6)
            matrix_df = pd.DataFrame(matrix)

            resultante_df = pd.DataFrame({
                "fase_norm": np.round(x_norm, 5),
                "Media": np.round(mean_y, 6),
                "DP": np.round(std_y, 6),
                "Media_+DP": np.round(mean_y + std_y, 6),
                "Media_-DP": np.round(mean_y - std_y, 6),
            })

            st.dataframe(matrix_df.head(8), use_container_width=True, hide_index=True)
            st.caption(f"{N_NORM} pontos × {len(cycles_seg) + 4} colunas")

            dl1, dl2 = st.columns(2)
            with dl1:
                buf_matrix = BytesIO()
                with pd.ExcelWriter(buf_matrix, engine="openpyxl") as writer:
                    matrix_df.to_excel(writer, sheet_name="Matriz", index=False)
                    stats_df = pd.DataFrame({
                        "Ciclo": [cyc["label"] for cyc in cycles_seg] + ["MÉDIA", "DP"],
                        "Duração (ms)": [round(d, 3) for d in durations] + [round(mean_dur, 3), round(std_dur, 3)],
                    })
                    stats_df.to_excel(writer, sheet_name="Durações", index=False)
                st.download_button(
                    "⬇️ Matriz completa (Excel)",
                    buf_matrix.getvalue(),
                    "matriz_ciclos.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="sts_dl_matrix",
                )
            with dl2:
                buf_res = BytesIO()
                with pd.ExcelWriter(buf_res, engine="openpyxl") as writer:
                    resultante_df.to_excel(writer, sheet_name="Resultante", index=False)
                st.download_button(
                    "⬇️ Só a resultante (Excel)",
                    buf_res.getvalue(),
                    "resultante.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    key="sts_dl_resultante",
                )

            # ── Salvar resultado desta pessoa ───────────────────────────────
            st.divider()
            st.subheader("💾 Acumular resultados")
            person_name = st.text_input(
                "Nome desta pessoa (para identificar no Excel)",
                value=uploaded.name.rsplit(".", 1)[0] if uploaded else "Pessoa",
                key="sts_person_name",
            )
            if st.button("💾 Salvar resultado desta pessoa", use_container_width=True, type="primary", key="sts_save_person"):
                already = [r["name"] for r in st.session_state.sts_saved_results]
                if person_name in already:
                    st.session_state.sts_saved_results = [
                        r for r in st.session_state.sts_saved_results if r["name"] != person_name
                    ]
                st.session_state.sts_saved_results.append({
                    "name": person_name,
                    "resultante_df": resultante_df.copy(),
                    "mean_dur": mean_dur,
                    "std_dur": std_dur,
                    "n_cycles": len(cycles_seg),
                })
                st.success(f"✅ '{person_name}' salvo! Total acumulado: {len(st.session_state.sts_saved_results)} pessoa(s).")

    # ── Painel de resultados acumulados ─────────────────────────────────────
    if st.session_state.sts_saved_results:
        st.divider()
        st.header(f"👥 Resultados acumulados — {len(st.session_state.sts_saved_results)} pessoa(s)")

        summary = pd.DataFrame([{
            "Nome": r["name"],
            "Nº ciclos": r["n_cycles"],
            "Duração média (ms)": round(r["mean_dur"], 1),
            "DP (ms)": round(r["std_dur"], 1),
            "CV (%)": round(r["std_dur"] / r["mean_dur"] * 100, 1) if r["mean_dur"] else 0,
        } for r in st.session_state.sts_saved_results])
        st.dataframe(summary, use_container_width=True, hide_index=True)

        if len(st.session_state.sts_saved_results) >= 2:
            st.subheader("📈 Resultantes individuais + curva geral do grupo")

            PALETTE2 = [
                "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
                "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
                "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
                "#4C78A8", "#F58518", "#E45756", "#72B7B2", "#54A24B",
            ]

            N_G = 300
            x_g = np.linspace(0, 1, N_G)
            all_medias = []
            fig_grupo = go.Figure()

            for i, r in enumerate(st.session_state.sts_saved_results):
                rdf = r["resultante_df"]
                fase = rdf["fase_norm"].values
                media_ind = rdf["Media"].values
                y_interp = np.interp(x_g, fase, media_ind)
                all_medias.append(y_interp)
                fig_grupo.add_trace(go.Scatter(
                    x=x_g, y=y_interp,
                    mode="lines", name=r["name"],
                    line=dict(color=PALETTE2[i % len(PALETTE2)], width=1.5),
                    opacity=0.6,
                ))

            grand_mean = np.mean(all_medias, axis=0)
            grand_std = np.std(all_medias, axis=0)

            fig_grupo.add_trace(go.Scatter(
                x=np.concatenate([x_g, x_g[::-1]]),
                y=np.concatenate([grand_mean + grand_std, (grand_mean - grand_std)[::-1]]),
                fill="toself", fillcolor="rgba(0,0,0,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="±1 DP geral", showlegend=True,
            ))
            fig_grupo.add_trace(go.Scatter(
                x=x_g, y=grand_mean,
                mode="lines", name="Média geral",
                line=dict(color="black", width=3),
            ))

            fig_grupo.update_layout(
                width=650, height=650,
                title="Resultantes individuais e curva geral do grupo",
                xaxis_title="Fase normalizada",
                yaxis_title=y_col if y_col else "Sinal",
                hovermode="x unified",
                plot_bgcolor="white", paper_bgcolor="white",
                legend=dict(orientation="h", y=-0.25),
                margin=dict(l=10, r=10, t=50, b=100),
            )
            fig_grupo.update_xaxes(showgrid=True, gridcolor="#eee")
            fig_grupo.update_yaxes(showgrid=True, gridcolor="#eee")
            st.plotly_chart(fig_grupo, use_container_width=False, key="sts_fig_grupo")

            grand_df = pd.DataFrame({
                "fase_norm": np.round(x_g, 5),
                "Media_geral": np.round(grand_mean, 6),
                "DP_geral": np.round(grand_std, 6),
                "Media_+DP": np.round(grand_mean + grand_std, 6),
                "Media_-DP": np.round(grand_mean - grand_std, 6),
            })
            buf_grand = BytesIO()
            with pd.ExcelWriter(buf_grand, engine="openpyxl") as writer:
                grand_df.to_excel(writer, sheet_name="Resultante_Geral", index=False)
            st.download_button(
                "⬇️ Exportar resultante geral do grupo (Excel)",
                buf_grand.getvalue(),
                "resultante_geral.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key="sts_dl_grand",
            )

        col_exp, col_clear2 = st.columns([2, 1])
        with col_exp:
            buf_all = BytesIO()
            with pd.ExcelWriter(buf_all, engine="openpyxl") as writer:
                summary.to_excel(writer, sheet_name="Resumo", index=False)
                for r in st.session_state.sts_saved_results:
                    sheet_name = r["name"][:31]
                    r["resultante_df"].to_excel(writer, sheet_name=sheet_name, index=False)
            st.download_button(
                "⬇️ Exportar todos os resultados (Excel)",
                buf_all.getvalue(),
                "resultados_grupo.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                type="primary",
                key="sts_dl_all",
            )
        with col_clear2:
            if st.button("🗑️ Limpar acumulados", use_container_width=True, key="sts_clear_acc"):
                st.session_state.sts_saved_results = []
                st.rerun()
