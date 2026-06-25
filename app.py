"""
MedStatDoc — Blood Report Interpreter via Hypothesis Testing
=============================================================
A Streamlit application that applies Z-tests, t-tests, and
confidence intervals to blood report values. Uses Gaussian Naive
Bayes for overall health risk prediction.

Launch:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

from blood_parameters import (
    BloodParameter,
    get_parameters_for_gender,
    get_parameter_categories,
)
from statistical_engine import run_analysis, Status
from naive_bayes_classifier import HealthRiskClassifier
from utils import (
    generate_explanation,
    format_p_value,
    get_status_emoji,
    get_status_color,
    get_risk_color,
    get_risk_emoji,
)


# ══════════════════════════════════════════════════════════════════════════════
# Page Config & Styling
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MedStatDoc — Blood Report Interpreter",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Initialize Session State
# ══════════════════════════════════════════════════════════════════════════════

if "classifier" not in st.session_state:
    st.session_state.classifier = HealthRiskClassifier()

if "results" not in st.session_state:
    st.session_state.results = None

if "risk_prediction" not in st.session_state:
    st.session_state.risk_prediction = None


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-header">
            <h2>🩺 MedStatDoc</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Configuration")

    gender = st.radio(
        "**Biological Sex**",
        options=["Male", "Female"],
        index=0,
        help="Affects gender-specific normal ranges (e.g., hemoglobin, RBC count).",
    )

    confidence_level = st.radio(
        "**Primary Confidence Level**",
        options=["95% (α = 0.05)", "99% (α = 0.01)"],
        index=0,
        help="The confidence level for hypothesis testing. 95% is standard; 99% is stricter.",
    )

    test_mode = st.radio(
        "**Test Mode**",
        options=["Single Value (Z-test)", "Multiple Readings (t-test)"],
        index=0,
        help="Single Value uses Z-test with known population σ. Multiple Readings uses t-test.",
    )

    st.markdown("---")

    # ── About Section ─────────────────────────────────────────────────────
    with st.expander("📖 About the Methodology", expanded=False):
        st.markdown("""
        **MedStatDoc** applies statistical hypothesis testing to your blood
        report values:

        **Z-Test** (Single Value)
        - Tests whether your value significantly differs from the population mean
        - H₀: Your value = Population mean
        - H₁: Your value ≠ Population mean
        - Uses known population standard deviation

        **t-Test** (Multiple Readings)
        - Same hypotheses but uses sample standard deviation
        - Better when you have multiple readings from different tests

        **Classification**
        - 🟢 **Normal:** p > 0.05
        - 🟡 **Borderline:** 0.01 < p ≤ 0.05
        - 🔴 **Flagged:** p ≤ 0.01

        **Naive Bayes Classifier**
        - Estimates overall health risk from your z-score profile
        - Trained on synthetic data from medical distributions
        """)

    with st.expander("⚠️ Disclaimer", expanded=False):
        st.markdown("""
        This tool provides **statistical analysis only**. It is **NOT** a
        medical diagnosis. Always consult a qualified healthcare professional
        for medical advice. Statistical significance does not necessarily
        imply clinical significance.
        """)


# ══════════════════════════════════════════════════════════════════════════════
# Main Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class="main-header">
        <h1>🩺 MedStatDoc</h1>
        <p class="subtitle">
            Blood Report Interpreter powered by Statistical Hypothesis Testing
            &amp; Naive Bayes Classification
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Get parameters for selected gender ────────────────────────────────────
params = get_parameters_for_gender(gender.lower())
categories = get_parameter_categories(params)

# Category icons mapping
CATEGORY_ICONS = {
    "Hematology": "🔬",
    "Metabolic": "⚡",
    "Lipid Panel": "💧",
    "Endocrine": "🧬",
    "Vitamins": "☀️",
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def _render_bell_curve(result):
    """Render a small normal distribution curve showing where the patient value falls."""
    mu = result.population_mean
    sigma = result.population_std
    x_val = result.patient_value
    color = get_status_color(result.status)

    # Generate curve points
    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    from scipy.stats import norm
    y_range = norm.pdf(x_range, mu, sigma)

    fig = go.Figure()

    # ── Fill regions ──────────────────────────────────────────────────
    # Normal (within 1.96σ)
    mask_normal = (x_range >= mu - 1.96 * sigma) & (x_range <= mu + 1.96 * sigma)
    fig.add_trace(go.Scatter(
        x=x_range[mask_normal],
        y=y_range[mask_normal],
        fill='tozeroy',
        fillcolor='rgba(16, 185, 129, 0.15)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))

    # Full curve
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        line=dict(color='rgba(148, 163, 184, 0.5)', width=2),
        showlegend=False,
        hoverinfo='skip',
    ))

    # Patient value marker
    y_patient = norm.pdf(x_val, mu, sigma)
    fig.add_trace(go.Scatter(
        x=[x_val],
        y=[y_patient],
        mode='markers+text',
        marker=dict(size=12, color=color, symbol='diamond'),
        text=[f"  {x_val:.1f}"],
        textposition='top right',
        textfont=dict(size=12, color=color, family='JetBrains Mono'),
        showlegend=False,
        hovertemplate=f"Your value: {x_val:.2f}<extra></extra>",
    ))

    # Vertical line at patient value
    fig.add_vline(x=x_val, line_dash="dash", line_color=color, line_width=1.5)
    # Mean line
    fig.add_vline(x=mu, line_dash="dot", line_color="rgba(255,255,255,0.3)", line_width=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
        margin=dict(l=20, r=20, t=10, b=30),
        xaxis=dict(
            title=dict(text=f"{result.unit}", font=dict(size=11)),
            showgrid=False,
        ),
        yaxis=dict(visible=False),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"bell_{result.parameter_name}_{id(result)}")


# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════

tab_input, tab_results, tab_risk = st.tabs([
    "📋 Enter Blood Report",
    "📊 Results Dashboard",
    "🎯 Health Risk Assessment",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: Input Form
# ──────────────────────────────────────────────────────────────────────────────

with tab_input:
    st.markdown(
        """
        <div class="glass-card">
            <p style="color: #94a3b8; margin: 0; line-height: 1.6;">
                Enter your blood test values below. Leave fields at <strong>0.0</strong>
                to skip parameters you don't have results for. Values are tested against
                population normal ranges using statistical hypothesis testing.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    is_t_test = test_mode.startswith("Multiple")

    with st.form("blood_report_form", clear_on_submit=False):
        input_values = {}

        for cat_name, cat_keys in categories.items():
            icon = CATEGORY_ICONS.get(cat_name, "📌")
            st.markdown(
                f'<div class="category-header">{icon} {cat_name}</div>',
                unsafe_allow_html=True,
            )

            cols = st.columns(min(len(cat_keys), 3))

            for i, key in enumerate(cat_keys):
                param = params[key]
                col = cols[i % len(cols)]

                with col:
                    range_str = f"{param.normal_range_low} – {param.normal_range_high}"
                    label = f"**{param.name}** ({param.unit})"
                    help_text = (
                        f"{param.description}\n\n"
                        f"Normal: {range_str} {param.unit} | "
                        f"μ = {param.population_mean}, σ = {param.population_std}"
                    )

                    if is_t_test:
                        raw = st.text_input(
                            label,
                            value="",
                            placeholder=f"e.g. 14.2, 14.5, 14.0",
                            help=help_text + "\n\nEnter comma-separated readings for t-test.",
                            key=f"input_{key}",
                        )
                        if raw.strip():
                            try:
                                parsed = [float(v.strip()) for v in raw.split(",") if v.strip()]
                                if parsed:
                                    input_values[key] = parsed
                            except ValueError:
                                st.error(f"Invalid input for {param.name}")
                    else:
                        val = st.number_input(
                            label,
                            min_value=0.0,
                            max_value=99999.0,
                            value=0.0,
                            step=0.1,
                            format="%.2f",
                            help=help_text,
                            key=f"input_{key}",
                        )
                        if val > 0:
                            input_values[key] = val

            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Submit Button ─────────────────────────────────────────────────
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            submitted = st.form_submit_button(
                "🔬 Analyze Blood Report",
                use_container_width=True,
                type="primary",
            )

    # ── Process form ──────────────────────────────────────────────────────
    if submitted:
        if not input_values:
            st.warning("⚠️ Please enter at least one blood test value to analyze.")
        else:
            with st.spinner("Running statistical analysis..."):
                results = run_analysis(input_values, params)
                st.session_state.results = results

                # Run Naive Bayes prediction
                prediction = st.session_state.classifier.predict(results)
                st.session_state.risk_prediction = prediction

            st.success(
                f"✅ Analysis complete — **{len(results)} parameters** tested. "
                f"Switch to the **Results Dashboard** tab to view findings."
            )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: Results Dashboard
# ──────────────────────────────────────────────────────────────────────────────

with tab_results:
    results = st.session_state.results

    if results is None:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 3rem;">
                <p style="font-size: 3rem; margin-bottom: 0.5rem;">📋</p>
                <p style="color: #94a3b8; font-size: 1.1rem;">
                    No results yet. Enter your blood test values in the
                    <strong>Enter Blood Report</strong> tab and click
                    <strong>Analyze</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # ── Summary Metrics ───────────────────────────────────────────────
        n_total = len(results)
        n_normal = sum(1 for r in results if r.status == Status.NORMAL)
        n_borderline = sum(1 for r in results if r.status == Status.BORDERLINE)
        n_flagged = sum(1 for r in results if r.status == Status.FLAGGED)

        cols = st.columns(4)
        with cols[0]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #3b82f6;">{n_total}</div>
                    <div class="metric-label">Tests Run</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[1]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #10b981;">{n_normal}</div>
                    <div class="metric-label">Normal</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[2]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #f59e0b;">{n_borderline}</div>
                    <div class="metric-label">Borderline</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with cols[3]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #ef4444;">{n_flagged}</div>
                    <div class="metric-label">Flagged</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Results Table ─────────────────────────────────────────────────
        st.markdown("### 📊 Detailed Results")

        # Build dataframe
        table_data = []
        for r in results:
            emoji = get_status_emoji(r.status)
            ci_str = f"[{r.ci_95[0]:.2f}, {r.ci_95[1]:.2f}]"
            table_data.append({
                "Status": f"{emoji} {r.status.value}",
                "Parameter": r.parameter_name,
                "Your Value": f"{r.patient_value:.2f} {r.unit}",
                "Normal Range": f"{r.normal_range_low} – {r.normal_range_high}",
                "Test": r.test_type,
                "Z/t Score": f"{r.test_statistic:.3f}",
                "P-Value": format_p_value(r.p_value),
                "95% CI": ci_str,
            })

        df = pd.DataFrame(table_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Status": st.column_config.TextColumn(width="small"),
                "P-Value": st.column_config.TextColumn(width="small"),
            },
        )

        # ── Download CSV ──────────────────────────────────────────────────
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="medstatdoc_results.csv",
            mime="text/csv",
        )

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Z-Score Visualization ─────────────────────────────────────────
        st.markdown("### 📈 Z-Score Distribution")

        fig_z = go.Figure()

        # Add reference bands
        fig_z.add_hrect(
            y0=-1.96, y1=1.96,
            fillcolor="rgba(16, 185, 129, 0.08)",
            line_width=0,
            annotation_text="95% Normal Range",
            annotation_position="top left",
        )
        fig_z.add_hrect(
            y0=-2.576, y1=-1.96,
            fillcolor="rgba(245, 158, 11, 0.08)",
            line_width=0,
        )
        fig_z.add_hrect(
            y0=1.96, y1=2.576,
            fillcolor="rgba(245, 158, 11, 0.08)",
            line_width=0,
        )

        # Add reference lines
        fig_z.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        fig_z.add_hline(y=1.96, line_dash="dash", line_color="rgba(245, 158, 11, 0.4)")
        fig_z.add_hline(y=-1.96, line_dash="dash", line_color="rgba(245, 158, 11, 0.4)")
        fig_z.add_hline(y=2.576, line_dash="dash", line_color="rgba(239, 68, 68, 0.4)")
        fig_z.add_hline(y=-2.576, line_dash="dash", line_color="rgba(239, 68, 68, 0.4)")

        # Add data points
        colors = [get_status_color(r.status) for r in results]
        names = [r.parameter_name for r in results]
        z_scores = [r.test_statistic for r in results]

        fig_z.add_trace(go.Bar(
            x=names,
            y=z_scores,
            marker_color=colors,
            marker_line_color=colors,
            marker_line_width=1,
            text=[f"{z:.2f}" for z in z_scores],
            textposition="outside",
            textfont=dict(size=11, color="#94a3b8"),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Z-score: %{y:.3f}<br>"
                "<extra></extra>"
            ),
        ))

        fig_z.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            height=450,
            margin=dict(l=40, r=40, t=20, b=80),
            xaxis=dict(
                tickangle=-45,
                gridcolor="rgba(255,255,255,0.05)",
            ),
            yaxis=dict(
                title="Z-Score (σ from mean)",
                gridcolor="rgba(255,255,255,0.05)",
                zeroline=True,
                zerolinecolor="rgba(255,255,255,0.1)",
            ),
            showlegend=False,
        )

        st.plotly_chart(fig_z, use_container_width=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Detailed Explanations ─────────────────────────────────────────
        st.markdown("### 📝 Detailed Explanations")

        # Show flagged first, then borderline, then normal
        sorted_results = sorted(
            results,
            key=lambda r: (
                0 if r.status == Status.FLAGGED else
                1 if r.status == Status.BORDERLINE else 2
            ),
        )

        for r in sorted_results:
            emoji = get_status_emoji(r.status)
            color = get_status_color(r.status)
            status_class = {
                Status.NORMAL: "status-normal",
                Status.BORDERLINE: "status-borderline",
                Status.FLAGGED: "status-flagged",
            }[r.status]

            with st.expander(
                f"{emoji} {r.parameter_name} — {r.patient_value:.2f} {r.unit} "
                f"(p = {format_p_value(r.p_value)})",
                expanded=(r.status != Status.NORMAL),
            ):
                explanation = generate_explanation(r)
                st.markdown(explanation)

                # Mini visualization — bell curve with patient value
                _render_bell_curve(r)



# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: Health Risk Assessment
# ──────────────────────────────────────────────────────────────────────────────

with tab_risk:
    prediction = st.session_state.risk_prediction

    if prediction is None:
        st.markdown(
            """
            <div class="glass-card" style="text-align: center; padding: 3rem;">
                <p style="font-size: 3rem; margin-bottom: 0.5rem;">🎯</p>
                <p style="color: #94a3b8; font-size: 1.1rem;">
                    Health risk assessment will appear here after you analyze
                    your blood report.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        risk = prediction["risk_category"]
        probs = prediction["probabilities"]
        contributions = prediction["feature_contributions"]

        risk_emoji = get_risk_emoji(risk)
        risk_color = get_risk_color(risk)

        risk_class = {
            "Low Risk": "risk-low",
            "Moderate Risk": "risk-moderate",
            "High Risk": "risk-high",
        }.get(risk, "risk-low")

        # ── Main Risk Card ────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="risk-card {risk_class} fade-in pulse-glow">
                <div style="font-size: 3.5rem; margin-bottom: 0.5rem;">{risk_emoji}</div>
                <div class="risk-title" style="color: {risk_color};">
                    {risk}
                </div>
                <div class="risk-subtitle">
                    Based on Gaussian Naive Bayes classification of your z-score profile
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Probability Distribution ──────────────────────────────────────
        st.markdown("### 📊 Risk Probability Distribution")

        prob_colors = {
            "Low Risk": "#10b981",
            "Moderate Risk": "#f59e0b",
            "High Risk": "#ef4444",
        }

        fig_prob = go.Figure()
        for label, prob in probs.items():
            fig_prob.add_trace(go.Bar(
                x=[label],
                y=[prob * 100],
                marker_color=prob_colors.get(label, "#6b7280"),
                marker_line_color=prob_colors.get(label, "#6b7280"),
                marker_line_width=1.5,
                text=[f"{prob*100:.1f}%"],
                textposition="outside",
                textfont=dict(size=14, family="JetBrains Mono", color="#e2e8f0"),
                showlegend=False,
                hovertemplate=f"<b>{label}</b><br>Probability: {prob*100:.1f}%<extra></extra>",
            ))

        fig_prob.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8"),
            height=350,
            margin=dict(l=40, r=40, t=20, b=40),
            yaxis=dict(
                title="Probability (%)",
                range=[0, 110],
                gridcolor="rgba(255,255,255,0.05)",
            ),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            showlegend=False,
        )

        st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Feature Contributions ─────────────────────────────────────────
        st.markdown("### 🔍 Parameter Impact Analysis")
        st.markdown(
            """
            <p style="color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem;">
                Which blood parameters contributed most to the risk assessment,
                ranked by absolute z-score magnitude.
            </p>
            """,
            unsafe_allow_html=True,
        )

        if contributions:
            impact_colors = {
                "High Impact": "#ef4444",
                "Moderate Impact": "#f59e0b",
                "Low Impact": "#3b82f6",
                "Minimal": "#6b7280",
            }

            fig_impact = go.Figure()

            param_names = [c[0] for c in contributions]
            z_scores_contrib = [abs(c[1]) for c in contributions]
            bar_colors = [impact_colors.get(c[2], "#6b7280") for c in contributions]
            impact_labels = [c[2] for c in contributions]

            fig_impact.add_trace(go.Bar(
                y=param_names[::-1],
                x=z_scores_contrib[::-1],
                orientation='h',
                marker_color=bar_colors[::-1],
                text=[f"|z| = {z:.2f} — {lbl}" for z, lbl in zip(
                    z_scores_contrib[::-1], impact_labels[::-1]
                )],
                textposition='outside',
                textfont=dict(size=11, family="JetBrains Mono", color="#94a3b8"),
                hovertemplate="<b>%{y}</b><br>|Z-score|: %{x:.3f}<extra></extra>",
            ))

            # Reference lines
            fig_impact.add_vline(x=1.96, line_dash="dash",
                                line_color="rgba(245, 158, 11, 0.5)", line_width=1,
                                annotation_text="α=0.05", annotation_position="top right")
            fig_impact.add_vline(x=2.576, line_dash="dash",
                                line_color="rgba(239, 68, 68, 0.5)", line_width=1,
                                annotation_text="α=0.01", annotation_position="top right")

            fig_impact.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8"),
                height=max(300, len(contributions) * 50),
                margin=dict(l=120, r=120, t=20, b=40),
                xaxis=dict(
                    title="Absolute Z-Score",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                showlegend=False,
            )

            st.plotly_chart(fig_impact, use_container_width=True)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        # ── Disclaimer ────────────────────────────────────────────────────
        st.markdown(
            """
            <div class="disclaimer">
                <strong>⚕️ Important Disclaimer:</strong> MedStatDoc provides
                <strong>statistical analysis only</strong>. The risk categories are
                based on mathematical models, not clinical judgment. Statistical
                significance does not necessarily imply clinical significance.
                <strong>Always consult a qualified healthcare professional</strong>
                for medical advice, diagnosis, or treatment decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div style="text-align: center; padding: 2rem 0 1rem; color: #475569; font-size: 0.8rem;">
        MedStatDoc v1.0 — Statistical Hypothesis Testing &amp; Naive Bayes Classification<br>
        Curriculum: Probability &amp; Statistics (MA136) — Z-test, t-test, Confidence Intervals, Point Estimation
    </div>
    """,
    unsafe_allow_html=True,
)
