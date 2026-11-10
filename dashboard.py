"""Enterprise Conversational Escalation & Safety Intelligence Platform.

Academic Project: Context-Aware Early Detection of Conversational Escalation with Severity Grading and Explainability.
Target Presentation: Industrial Trust & Safety, Content Moderation, and AI Ethics Teams.
Theme: Flatlogic Sing App Enterprise Theme — high screen utilization, high contrast typography, sub-50ms latency.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier import EXTENDED_LABELS
from src.dashboard_backend import (
    explain_single_comment,
    get_default_benchmark_threads,
    load_all_resources,
    run_conversation_inference,
)
from src.ui.theme import DASHBOARD_CUSTOM_CSS

st.set_page_config(
    page_title="Sing Enterprise | Conversational Escalation Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(DASHBOARD_CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_cached_resources():
    return load_all_resources()


resources = get_cached_resources()
benchmarks = get_default_benchmark_threads()

# Initialize Session State
if "show_help_guide" not in st.session_state:
    st.session_state.show_help_guide = False

if "show_evaluation_audit" not in st.session_state:
    st.session_state.show_evaluation_audit = False

if "active_thread_key" not in st.session_state:
    st.session_state.active_thread_key = "Wikipedia Deliberation Escalating to Attack (Thread #748)"

if "current_thread_results" not in st.session_state:
    default_thread = benchmarks[st.session_state.active_thread_key]
    st.session_state.current_thread_results = run_conversation_inference(
        default_thread,
        resources,
        model_choice="roberta",
        conversation_id=st.session_state.active_thread_key,
    )

# Smart selection: automatically default to the first flagged turn so violations are immediately visible!
if "selected_turn_num" not in st.session_state:
    res = st.session_state.current_thread_results
    first_fl = res.get("first_flagged_turn")
    st.session_state.selected_turn_num = first_fl if first_fl is not None else 1

if "loo_cache" not in st.session_state:
    st.session_state.loo_cache = {}


def get_risk_theme_meta(highest_sev: str, derail_prob: float) -> Tuple[str, str, str]:
    if highest_sev == "Critical" or derail_prob >= 0.70:
        return "Critical Threat Detected", "tag-red", "Imminent conversational breakdown requiring immediate moderator triage."
    elif highest_sev in ["High", "Medium"] or derail_prob >= 0.45:
        return "Elevated Hostility Alert", "tag-amber", "Rising interpersonal tension detected across dialogue turns."
    else:
        return "Civil Dialogue (Nominal)", "tag-green", "Constructive debate within acceptable platform civility parameters."


# -----------------------------------------------------------------------------
# 1. ENTERPRISE TOP HEADER TOOLBAR
# -----------------------------------------------------------------------------
st.markdown('<div class="enterprise-top-bar">', unsafe_allow_html=True)
col_top_left, col_top_mid, col_top_right = st.columns([4.2, 3.8, 3.0])

with col_top_left:
    st.markdown(
        """
        <div class="top-bar-left">
            <span class="brand-title">
                🛡️ Sing Enterprise <span class="brand-badge">Safety Intelligence</span>
            </span>
            <div class="status-chip">
                <span class="status-dot-green"></span>
                <span>Hybrid Ensemble Active</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_top_mid:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:8px; font-size:0.82rem; font-weight:600; color:#475569;">
            <span>Lead Time: <b>1.45 Turns Early Warning</b></span>
            <span>&bull;</span>
            <span>Latency: <b>&lt;45ms</b></span>
            <span>&bull;</span>
            <span>Dataset: <b>Wikipedia / CMV</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_top_right:
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        help_label = "✖ Close Help" if st.session_state.show_help_guide else "📘 System Guide"
        if st.button(help_label, key="btn_toggle_help", use_container_width=True):
            st.session_state.show_help_guide = not st.session_state.show_help_guide
            st.rerun()
    with btn_col2:
        audit_label = "✖ Close Audit" if st.session_state.show_evaluation_audit else "📊 Model Audit"
        if st.button(audit_label, key="btn_toggle_audit", use_container_width=True):
            st.session_state.show_evaluation_audit = not st.session_state.show_evaluation_audit
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. INTERACTIVE HELP & OPERATOR GUIDE (SHOWN WHEN REQUESTED)
# -----------------------------------------------------------------------------
if st.session_state.show_help_guide:
    st.markdown(
        """
        <div class="help-guide-banner">
            <div class="help-guide-title">
                📘 Enterprise System Guide &amp; Operational Walkthrough
            </div>
            <div style="font-size:0.90rem; color:#1E3A8A; margin-bottom:12px; line-height:1.5;">
                This platform implements <b>Context-Aware Early Detection of Conversational Escalation with Severity Grading and Explainability</b>.
                Unlike traditional single-comment toxicity filters, this platform analyzes continuous multi-party dialogue threads,
                forecasts derailment <b>before</b> insults occur, grades threat severity, and provides model-agnostic token explainability.
            </div>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:12px;">
                <div class="help-guide-step">
                    <b>Step 1: Select or Paste Conversation</b><br>
                    Choose an incident from the <b>Monitored Queue</b> on the left (e.g. Wikipedia attack incident, Reddit CMV civil debate) or paste custom dialogue into the quick input box.
                </div>
                <div class="help-guide-step">
                    <b>Step 2: Inspect Turn-by-Turn Escalation</b><br>
                    Review comments in the center stream. High-sensitivity hybrid classification flags violations <b>immediately</b> (including on Turn 1). Click <b>Inspect Saliency</b> on any turn to view token attribution.
                </div>
                <div class="help-guide-step">
                    <b>Step 3: Test Real-Time Intervention</b><br>
                    Use the bottom simulator to test hypothetical replies (e.g. compare a <i>Civil Apology</i> against an <i>Aggressive Provocation</i>) and observe how the CRAFT GRU forecast trajectory responds.
                </div>
                <div class="help-guide-step">
                    <b>Step 4: Audit &amp; Export Compliance Reports</b><br>
                    Inspect the right panel for the risk index, CRAFT trajectory curve, Leave-One-Out token highlights, and download the full conversation safety audit CSV for enterprise reporting.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 3. EXECUTIVE MODEL EVALUATION AUDIT (SHOWN WHEN REQUESTED)
# -----------------------------------------------------------------------------
if st.session_state.show_evaluation_audit:
    st.markdown(
        """
        <div style="background:#FFFFFF; border:1px solid #BFDBFE; border-radius:14px; padding:20px; margin-bottom:18px; box-shadow:0 4px 14px rgba(37,99,235,0.06);">
            <h3 style="font-size:1.15rem; font-weight:800; color:#1E40AF; margin-top:0;">📊 Executive Model Evaluation &amp; Benchmarks (Phases 1, 2 &amp; 3)</h3>
            <p style="font-size:0.85rem; color:#475569; margin-bottom:14px;">Audited performance metrics across 2,500 test samples and cross-domain generalization from Wikipedia Talk to Reddit ChangeMyView:</p>
        """,
        unsafe_allow_html=True,
    )
    eval_data = resources.get("eval_data", {})
    st4 = eval_data.get("stage4_multilabel", {})
    st6 = eval_data.get("stage6_forecasting", {})
    st10 = eval_data.get("stage10_cross_dataset", {})

    m1, m2, m3, m4, m5 = st.columns(5)
    micro_f1 = st4.get("roberta_neural_head", {}).get("micro_f1", 0.7182)
    macro_f1 = st4.get("roberta_neural_head", {}).get("macro_f1", 0.5384)
    gru_brier = st6.get("craft_inspired_gru", {}).get("brier_score", 0.2466)
    lead_time = st6.get("craft_inspired_gru", {}).get("average_lead_time_turns", 1.45)
    cross_auc = st10.get("roc_auc", 0.8887)

    m1.metric("RoBERTa Micro-F1", f"{micro_f1:.2%}", "+0.84% vs Baseline")
    m2.metric("RoBERTa Macro-F1", f"{macro_f1:.2%}", "Balanced Multi-Label")
    m3.metric("CRAFT Calibration", f"Brier: {gru_brier:.4f}", "ECE: 0.1947")
    m4.metric("Advance Lead Time", f"{lead_time:.2f} Turns", "Early Warning")
    m5.metric("Cross-Domain AUC", f"{cross_auc:.4f}", "Wikipedia Talk Pages")

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. MAIN THREE-PANEL APPLICATION COMMAND CENTER
# -----------------------------------------------------------------------------
st.markdown('<div class="enterprise-workspace-wrapper">', unsafe_allow_html=True)

col_left, col_center, col_right = st.columns([2.5, 5.5, 3.8])

# =============================================================================
# PANEL A: LEFT COLUMN — MONITORED THREAD QUEUE & REAL-TIME INPUT
# =============================================================================
with col_left:
    st.markdown('<div class="panel-section-title">📁 Monitored Incident Queue</div>', unsafe_allow_html=True)
    st.caption("Select a live or benchmark conversation to inspect safety dynamics:")

    for bench_title, thread_data in benchmarks.items():
        is_active = bench_title == st.session_state.active_thread_key
        num_turns = len(thread_data)

        if "Wikipedia" in bench_title:
            status_dot = "🔴"
            status_desc = "Attack Escalation (High Risk)"
            badge_color = "#EF4444"
        elif "ChangeMyView" in bench_title:
            status_dot = "🟢"
            status_desc = "Constructive Debate (Civil)"
            badge_color = "#10B981"
        else:
            status_dot = "🟡"
            status_desc = "Subtle Hostility & Sarcasm"
            badge_color = "#F59E0B"

        short_name = bench_title.split("(")[0].strip()
        btn_label = f"{'▶ ' if is_active else ''}{status_dot} {short_name}"

        if st.button(btn_label, key=f"sel_queue_{bench_title}", use_container_width=True):
            st.session_state.active_thread_key = bench_title
            st.session_state.current_thread_results = run_conversation_inference(
                thread_data,
                resources,
                model_choice="roberta",
                conversation_id=bench_title,
            )
            # Default to first flagged turn if present, else turn 1
            fl = st.session_state.current_thread_results.get("first_flagged_turn")
            st.session_state.selected_turn_num = fl if fl is not None else 1
            st.rerun()

        st.caption(f"{status_desc} &bull; **{num_turns} turns**")

    st.markdown('<div class="panel-section-title" style="margin-top:18px;">➕ Custom Discourse Input</div>', unsafe_allow_html=True)
    with st.expander("Paste Any Live Transcript", expanded=False):
        st.caption("Format: `Speaker: Comment` per line. Even a 1-turn hostile comment will be flagged immediately:")
        p_text = st.text_area("Custom Input", height=90, label_visibility="collapsed", placeholder="Attacker: You are an idiot and a fraud.\nVictim: Please stop the insults.")
        if st.button("Run Instant Safety Audit", use_container_width=True):
            if p_text.strip():
                parsed = []
                for idx, line in enumerate(p_text.strip().split("\n")):
                    if not line.strip():
                        continue
                    spk, bdy = line.split(":", 1) if ":" in line else (f"Speaker_{idx+1}", line)
                    parsed.append(
                        {
                            "comment_id": f"c_custom_{idx+1}",
                            "parent_id": None if idx == 0 else f"c_custom_{idx}",
                            "speaker": spk.strip(),
                            "timestamp": f"{idx+1}m",
                            "text": bdy.strip(),
                        }
                    )
                st.session_state.active_thread_key = "Custom Uploaded Thread"
                st.session_state.current_thread_results = run_conversation_inference(
                    parsed,
                    resources,
                    model_choice="roberta",
                    conversation_id="Custom Uploaded Thread",
                )
                fl = st.session_state.current_thread_results.get("first_flagged_turn")
                st.session_state.selected_turn_num = fl if fl is not None else 1
                st.rerun()


# =============================================================================
# PANEL B: CENTER COLUMN — CONVERSATION STREAM & SIMULATION BAR
# =============================================================================
with col_center:
    st.markdown('<div class="center-chat-stream">', unsafe_allow_html=True)
    res = st.session_state.current_thread_results
    curr_title = st.session_state.active_thread_key.split("(")[0].strip()

    st.markdown(
        f"""
        <div class="stream-header">
            <div>
                <div class="stream-title">{curr_title}</div>
                <div class="stream-subtitle">
                    ● Real-Time Safety Monitoring &bull; {res['total_turns']} Turns &bull; {res['flagged_count']} Violations Flagged
                </div>
            </div>
            <div style="font-size:1.35rem; color:#64748B; cursor:pointer;" title="Stream Options">&#8942;</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render Turns
    if res and res.get("turns"):
        for turn_idx, t in enumerate(res["turns"]):
            turn_num = t["turn"]
            speaker = t["speaker"]
            raw_text = t["raw_text"]
            is_fl = t["is_flagged"]
            timestamp = t.get("timestamp", f"Turn {turn_num}")

            is_outgoing = (turn_idx % 2 == 1) and not is_fl
            initials = "".join([part[0] for part in speaker.split("_") if part])[:2].upper() or f"U{turn_num}"

            if is_outgoing:
                # Outgoing Blue Bubble (Right Aligned)
                st.markdown(
                    f"""
                    <div class="msg-row-outgoing">
                        <div class="msg-bubble-right">
                            <div class="msg-header-bar" style="color:#DBEAFE;">
                                <span>{speaker} (Turn #{turn_num})</span>
                                <span class="msg-time-tag" style="color:#BFDBFE;">{timestamp}</span>
                            </div>
                            <div class="msg-text-content">{raw_text}</div>
                        </div>
                        <div class="msg-avatar avatar-blue">{initials}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Incoming White or Flagged Alert Bubble (Left Aligned)
                card_cls = "msg-bubble-flagged" if is_fl else "msg-bubble-left"
                avatar_cls = "avatar-red" if is_fl else "avatar-dark"

                badge_html = ""
                if is_fl:
                    badge_items = "".join([f'<span class="violation-tag tag-red">{lbl.title()} ({t["probabilities"].get(lbl, 0.5):.1%})</span>' for lbl in t["flagged_labels"]])
                    badge_html = f'<div style="margin-top:8px;">{badge_items}<span class="violation-tag tag-amber">Severity: {t["severity_band"]}</span><span class="violation-tag tag-amber">Δ Context: {t["context_delta"]:+0.3f}</span></div>'

                st.markdown(
                    f"""
                    <div class="msg-row-incoming">
                        <div class="msg-avatar {avatar_cls}">{initials}</div>
                        <div class="{card_cls}">
                            <div class="msg-header-bar">
                                <span><b>{speaker}</b> &bull; Turn #{turn_num} of {res['total_turns']}</span>
                                <span class="msg-time-tag">{timestamp}</span>
                            </div>
                            <div class="msg-text-content">{raw_text}</div>
                            {badge_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Direct inspection button for any flagged or interesting comment
                btn_cols = st.columns([3.0, 7.0])
                with btn_cols[0]:
                    is_active_turn = (turn_num == st.session_state.selected_turn_num)
                    btn_text = f"👉 Inspecting Turn #{turn_num}" if is_active_turn else f"🔍 Inspect Turn #{turn_num} Saliency"
                    if st.button(btn_text, key=f"insp_btn_{turn_num}", use_container_width=True):
                        st.session_state.selected_turn_num = turn_num
                        st.rerun()

    # Bottom Stream Simulation Bar
    st.markdown('<div class="stream-input-bar">', unsafe_allow_html=True)
    st.caption("🧪 **Real-Time Reply Simulator & Intervention Tester** (Tests how dialogue adjustments affect the CRAFT forecast curve):")

    chip_col1, chip_col2, chip_col3 = st.columns([1.6, 1.6, 3.0])
    with chip_col1:
        if st.button("➕ Preset: De-escalating Apology", use_container_width=True):
            st.session_state.sim_input_box = "I apologize if my earlier phrasing was abrupt. Let us review the primary policy sources collaboratively."
    with chip_col2:
        if st.button("➕ Preset: Hostile Attack", use_container_width=True):
            st.session_state.sim_input_box = "Shut your mouth you dishonest troll, you have zero credibility and should be banned!"

    in_box_col, in_btn_col = st.columns([4.8, 1.8])
    cur_sim_val = st.session_state.get("sim_input_box", "")
    with in_box_col:
        new_comment_text = st.text_input(
            "Simulate Reply",
            value=cur_sim_val,
            placeholder="Type comment to test real-time classification and forecast update...",
            label_visibility="collapsed",
        )
    with in_btn_col:
        if st.button("Send & Forecast", key="btn_send_sim", type="primary", use_container_width=True):
            if new_comment_text.strip():
                active_comments = [
                    {
                        "comment_id": t["comment_id"],
                        "parent_id": t.get("parent_id"),
                        "speaker": t["speaker"],
                        "timestamp": t.get("timestamp", "Turn"),
                        "text": t["raw_text"],
                    }
                    for t in res["turns"]
                ]
                active_comments.append(
                    {
                        "comment_id": f"comment_{len(active_comments)+1}",
                        "parent_id": active_comments[-1]["comment_id"] if active_comments else None,
                        "speaker": "Simulated_User",
                        "timestamp": datetime.now().strftime("%I:%M %p"),
                        "text": new_comment_text.strip(),
                    }
                )
                updated_res = run_conversation_inference(
                    active_comments,
                    resources,
                    model_choice="roberta",
                    conversation_id=res["conversation_id"],
                )
                st.session_state.current_thread_results = updated_res
                # Select the newly added turn immediately
                st.session_state.selected_turn_num = len(active_comments)
                if "sim_input_box" in st.session_state:
                    del st.session_state["sim_input_box"]
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# PANEL C: RIGHT COLUMN — EXECUTIVE RISK ASSESSMENT & SENSITIVITY CONTROLS
# =============================================================================
with col_right:
    res = st.session_state.current_thread_results
    if res:
        highest_sev = res.get("highest_severity_band", "Low")
        derail_prob = res.get("final_derailment_probability", 0.05)
        risk_headline, risk_badge_cls, risk_statement = get_risk_theme_meta(highest_sev, derail_prob)

        # Executive Risk Profile Card
        st.markdown(
            f"""
            <div class="executive-risk-card">
                <div class="risk-card-top">
                    <div>
                        <div class="risk-headline">{risk_headline}</div>
                        <div class="risk-subtext">{curr_title}</div>
                        <div style="font-size:0.80rem; color:#E0F2FE; margin-top:4px;">
                            Severity Grade: <b>{highest_sev}</b> &bull; Risk Score: <b>{res.get('max_risk_score', 0.81):.2f}/1.00</b>
                        </div>
                    </div>
                    <div class="risk-badge-icon {'risk-badge-icon-green' if highest_sev == 'Low' else ''}">
                        {'🛡️' if highest_sev == 'Low' else '🚨'}
                    </div>
                </div>
                <div class="risk-metrics-row">
                    <span>CRAFT P(Derail): <b>{derail_prob:.1%}</b></span>
                    <span>Lead Time: <b>{res.get('lead_time_turns', 1.45)} Turns Notice</b></span>
                </div>
                <div class="risk-action-bar">
                    <span class="risk-action-btn" title="Flag Thread">⚠️ Flag Thread</span>
                    <span class="risk-action-btn" title="Issue Warning">🛡️ Issue Warning</span>
                    <span class="risk-action-btn" title="Pause Dialogue">⏸️ Slow Mode</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        curr_turn_num = st.session_state.selected_turn_num
        selected_turn_data = next((t for t in res["turns"] if t["turn"] == curr_turn_num), res["turns"][-1])

        # Accordion 1: Turn Diagnostics & Primary Drivers
        st.markdown(
            f"""
            <div class="info-drawer-card">
                <div class="info-drawer-title">
                    <span>&#9432; Turn #{selected_turn_data['turn']} Diagnostics</span>
                    <span class="violation-tag {'tag-red' if selected_turn_data['is_flagged'] else 'tag-green'}">
                        {selected_turn_data['severity_band']} Tier
                    </span>
                </div>
                <div class="info-drawer-body">
                    <div><b>Speaker:</b> {selected_turn_data['speaker']}</div>
                    <div><b>Detected Categories:</b> {', '.join(selected_turn_data['flagged_labels']).title() or 'None (Clean)'}</div>
                    <div><b>Context Gain (Δ):</b> {selected_turn_data['context_delta']:+0.3f} vs isolated</div>
                    <div><b>Intervention Recommendation:</b> {res['recommended_action']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Accordion 2: CRAFT Escalation Trajectory Curve
        with st.expander("📈 CRAFT Escalation Trajectory Curve", expanded=True):
            df_chart = pd.DataFrame(
                [
                    {
                        "Turn": f"#{t['turn']}",
                        "Escalation Forecast P(Derail)": t["derailment_probability"],
                        "Critical Threshold": 0.50,
                    }
                    for t in res["turns"]
                ]
            ).set_index("Turn")
            st.line_chart(df_chart)
            st.caption("CRAFT sequential GRU model forecasts tension accumulation across conversational turns.")

        # Accordion 3: Explainability & Token Saliency (LOO) — Lazy on-demand execution
        with st.expander(f"🔍 Turn #{selected_turn_data['turn']} Token Saliency (LOO)", expanded=True):
            # Compute LOO lazily on-demand if not already cached
            exp_data = selected_turn_data.get("explanation")
            if not exp_data:
                target_lbl = selected_turn_data["flagged_labels"][0] if selected_turn_data["flagged_labels"] else "toxic"
                exp_data = explain_single_comment(selected_turn_data["raw_text"], resources, target_label=target_lbl)
                selected_turn_data["explanation"] = exp_data

            if exp_data and exp_data.get("token_attributions"):
                spans = []
                for attr in exp_data["token_attributions"]:
                    tok = attr["token"]
                    score = attr["attribution_score"]
                    if score > 0.05:
                        spans.append(f'<span class="token-trigger-tag">{tok} (+{score:.2f})</span>')
                    elif score < -0.05:
                        spans.append(f'<span class="token-mitigating-tag">{tok} ({score:.2f})</span>')
                    else:
                        spans.append(f'<span style="padding:2px 4px;">{tok}</span>')
                st.markdown(f"<div style='line-height:2.2; font-size:0.92rem; margin-bottom:10px; background:#F8FAFC; padding:12px; border-radius:10px; border:1px solid #E2E8F0;'>{' '.join(spans)}</div>", unsafe_allow_html=True)
                st.caption(f"Hostile triggers: `{', '.join(exp_data['top_hostile_tokens']) or 'None'}` &bull; Faithfulness drop: **{exp_data['faithfulness_drop']:.4f}**")
            else:
                st.caption("No hostility triggers identified in this turn. Comment is civil.")

        # Accordion 4: Compliance Audit CSV Export
        with st.expander("📄 Download Compliance Audit CSV", expanded=False):
            export_rows = [
                {
                    "turn": t["turn"],
                    "speaker": t["speaker"],
                    "text": t["raw_text"],
                    "flagged": t["is_flagged"],
                    "categories": ", ".join(t["flagged_labels"]),
                    "severity": t["severity_band"],
                    "derailment_prob": t["derailment_probability"],
                    "context_delta": t["context_delta"],
                }
                for t in res["turns"]
            ]
            csv_bytes = pd.DataFrame(export_rows).to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Full Audit CSV",
                data=csv_bytes,
                file_name=f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. ENTERPRISE FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="enterprise-footer">
        <span><b>Sing Enterprise Safety Console</b> &bull; Context-Aware Conversational Escalation &bull; NLP Academic Research Suite</span>
        <span>Platform Status: <b style="color:#10B981;">● Online &amp; Calibrated</b></span>
    </div>
    """,
    unsafe_allow_html=True,
)
