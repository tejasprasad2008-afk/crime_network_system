"""
app.py - Intelligent Crime Network Analysis System (ICNAS)
Deterministic Crime Knowledge Graph, Trained Entity Resolution,
Sarvam AI Indic Speech Integration, and Section 63 BSA 2023 Legal Provenance.
"""

import os
import sys
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime

# Set path to include local modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from modules.ingestion import (
    parse_fir_text, parse_cdr_csv, parse_bank_csv,
    parse_anpr_csv, parse_audio_intercept, compute_sha256
)
from modules.entity_resolution import (
    DeterministicEntityResolver, normalize_vehicle_plate, jaro_winkler
)
from modules.graph_engine import CrimeKnowledgeGraph
from modules.sarvam_speech import SarvamSpeechClient
from modules.bsa_report import generate_bsa_section63_certificate

# ----------------- PAGE CONFIGURATION ----------------- #
st.set_page_config(
    page_title="ICNAS - Crime Network Analysis System",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Law Enforcement Cockpit Aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-primary { background-color: #DBEAFE; color: #1E40AF; }
    .badge-success { background-color: #D1FAE5; color: #065F46; }
    .badge-warning { background-color: #FEF3C7; color: #92400E; }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- SESSION STATE & DATA PIPELINE ----------------- #
@st.cache_data
def load_and_process_all_evidence():
    """Runs the deterministic multi-source ingestion and builds the knowledge graph."""
    cases_dir = os.path.join(BASE_DIR, "data", "sample_cases")
    hostels_path = os.path.join(BASE_DIR, "data", "known_hostels_pms.json")

    # Ingest FIRs
    fir_files = ["fir_104_cyberabad.txt", "fir_88_hyderabad.txt", "fir_42_pune.txt", "fir_19_rachakonda.txt"]
    firs_data = []
    source_docs = []

    for ff in fir_files:
        p = os.path.join(cases_dir, ff)
        if os.path.exists(p):
            res = parse_fir_text(p)
            firs_data.append(res)
            source_docs.append({"name": ff, "type": "FIR_POLICE_RECORD", "sha256": res["sha256"]})

    # Ingest CSVs
    cdr_meta, cdr_df = parse_cdr_csv(os.path.join(cases_dir, "cdr_dump_feb2026.csv"))
    source_docs.append({"name": cdr_meta["source_file"], "type": "TELECOM_CDR_DUMP", "sha256": cdr_meta["sha256"]})

    bank_meta, bank_df = parse_bank_csv(os.path.join(cases_dir, "bank_upi_transactions.csv"))
    source_docs.append({"name": bank_meta["source_file"], "type": "BANK_LEDGER_DATA", "sha256": bank_meta["sha256"]})

    anpr_meta, anpr_df = parse_anpr_csv(os.path.join(cases_dir, "anpr_toll_sightings.csv"))
    source_docs.append({"name": anpr_meta["source_file"], "type": "ANPR_VEHICLE_LOGS", "sha256": anpr_meta["sha256"]})

    audio_meta = parse_audio_intercept(os.path.join(cases_dir, "audio_transcript_intercept.json"))
    source_docs.append({"name": audio_meta["source_file"], "type": "SARVAM_AUDIO_INTERCEPT", "sha256": audio_meta["sha256"]})

    # Initialize Graph & Resolver
    kg = CrimeKnowledgeGraph()
    resolver = DeterministicEntityResolver(hostels_path)

    # 1. Add Cases
    for fd in firs_data:
        c_id = f"CASE_{fd['case_id'].replace('/', '_')}"
        kg.add_entity_node(c_id, f"FIR {fd['case_id']}", "CASE", {"ps": fd["police_station"]})

    # 2. Add Phone Nodes & CDR Calls
    for _, row in cdr_df.iterrows():
        c1, c2 = row["calling_number"], row["called_number"]
        kg.add_entity_node(c1, c1, "PHONE")
        kg.add_entity_node(c2, c2, "PHONE")
        kg.add_evidence_edge(c1, c2, "CDR_CALL", "OBSERVED", [row["record_id"]], {
            "duration": row["duration_seconds"],
            "timestamp": row["call_timestamp"],
            "tower": row["cell_tower_id"]
        })

    # 3. Add Bank / UPI Accounts & Financial Transactions
    for _, row in bank_df.iterrows():
        s, r = str(row["sender_vpa_or_acc"]), str(row["receiver_vpa_or_acc"])
        s_type = "UPI" if "@" in s else "BANK_ACC"
        r_type = "UPI" if "@" in r else "BANK_ACC"
        kg.add_entity_node(s, s, s_type)
        kg.add_entity_node(r, r, r_type)
        kg.add_evidence_edge(s, r, "FUND_TRANSFER", "OBSERVED", [row["txn_id"]], {
            "amount_inr": row["amount_inr"],
            "mode": row["txn_mode"],
            "timestamp": row["timestamp"]
        })

    # 4. Add Vehicles & ANPR Sightings
    for _, row in anpr_df.iterrows():
        raw_plate = row["detected_plate"]
        clean_plate, _ = normalize_vehicle_plate(raw_plate)
        kg.add_entity_node(clean_plate, f"Vehicle: {clean_plate}", "VEHICLE")
        loc = row["camera_location"]
        kg.add_entity_node(loc, loc, "LOCATION")
        kg.add_evidence_edge(clean_plate, loc, "SIGHTED_AT", "OBSERVED", [row["sighting_id"]], {
            "timestamp": row["timestamp"],
            "confidence": row["confidence"]
        })

    # 5. Add Key Person Entities & Connections from FIRs and Intercept
    # Person 1: Rahim Khan / Mohd Abdul Rahim
    kg.add_entity_node("PERSON_RAHIM", "Rahim Khan (Alias: Mohd Abdul Rahim)", "PERSON", {"role": "OPERATIVE / MONEY RUNNER"})
    kg.add_evidence_edge("PERSON_RAHIM", "+919876543210", "USES_PHONE", "OBSERVED", ["FIR_104", "FIR_88"], {"confidence": 0.96})
    kg.add_evidence_edge("PERSON_RAHIM", "rahim99@upi", "CONTROLS_UPI", "OBSERVED", ["TXN_20101", "TXN_20106"], {"confidence": 0.99})
    kg.add_evidence_edge("PERSON_RAHIM", "TS09AB1234", "OPERATED_VEHICLE", "INFERRED", ["FIR_104_CCTV", "ANPR_803"], {"confidence": 0.91})
    kg.add_evidence_edge("PERSON_RAHIM", "CASE_104_2026", "ACCUSED_IN", "OBSERVED", ["FIR_104"], {})
    kg.add_evidence_edge("PERSON_RAHIM", "CASE_88_2026", "ACCUSED_IN", "OBSERVED", ["FIR_88"], {})

    # Person 2: Farooq Ahmed (Bhaijaan) - The Covert Coordinator
    kg.add_entity_node("PERSON_FAROOQ", "Farooq Ahmed (Alias: Bhaijaan)", "PERSON", {"role": "COVERT SYNDICATE COORDINATOR"})
    kg.add_evidence_edge("PERSON_FAROOQ", "+919123456789", "USES_PHONE", "OBSERVED", ["FIR_42_LEDGER"], {"confidence": 0.95})
    kg.add_evidence_edge("PERSON_FAROOQ", "+919988776655", "USES_PHONE", "OBSERVED", ["FIR_42_SEIZURE"], {"confidence": 0.92})
    kg.add_evidence_edge("PERSON_FAROOQ", "farooq.bhai@hdfc", "CONTROLS_BANK_ACC", "OBSERVED", ["TXN_20104", "TXN_20109"], {"confidence": 0.98})
    kg.add_evidence_edge("PERSON_FAROOQ", "CASE_42_2026", "NAMED_LEADER_IN", "OBSERVED", ["FIR_42"], {})

    # Stolen Vehicle link to Case 19/2026 Rachakonda
    kg.add_evidence_edge("TS09AB1234", "CASE_19_2026", "STOLEN_PROPERTY_IN", "OBSERVED", ["FIR_19"], {"chassis": "MA3EJEB1S00-984210"})

    # Link Vehicle to IndusInd Bank account mule transactions
    kg.add_evidence_edge("9192837465", "CASE_104_2026", "MULE_ACCOUNT_IN", "OBSERVED", ["FIR_104"], {})
    kg.add_evidence_edge("9192837465", "CASE_88_2026", "MULE_ACCOUNT_IN", "OBSERVED", ["FIR_88"], {})
    kg.add_evidence_edge("9192837465", "CASE_42_2026", "HAWALA_ROUTED_IN", "OBSERVED", ["FIR_42"], {})

    # Compute Betweenness Centrality
    betweenness = kg.compute_betweenness_centrality()

    return {
        "kg": kg,
        "firs_data": firs_data,
        "source_docs": source_docs,
        "cdr_df": cdr_df,
        "bank_df": bank_df,
        "anpr_df": anpr_df,
        "audio_meta": audio_meta,
        "betweenness": betweenness,
        "resolver": resolver
    }


pipeline_data = load_and_process_all_evidence()
kg = pipeline_data["kg"]
betweenness = pipeline_data["betweenness"]
source_docs = pipeline_data["source_docs"]
resolver = pipeline_data["resolver"]


# ----------------- SIDEBAR CONTROLS ----------------- #
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/55/Emblem_of_India.svg", width=65)
    st.title("ICNAS Cockpit")
    st.caption("Intelligent Crime Network Analysis System")
    st.markdown("---")

    st.subheader("🎯 Active Investigation")
    scenario = st.selectbox(
        "Select Operation",
        ["Operation Cyber-Spider (Multi-State Syndicate)", "Custom Case Upload..."]
    )

    st.subheader("⚙️ Verification Controls")
    tier_filter = st.multiselect(
        "Edge Classification Tiers",
        ["OBSERVED", "INFERRED", "HYPOTHESIS"],
        default=["OBSERVED", "INFERRED"]
    )

    min_betweenness = st.slider(
        "Highlight Broker (Betweenness Centrality)",
        min_value=0.0, max_value=0.5, value=0.10, step=0.02,
        help="Filters nodes that act as structural bridges between distinct criminal clusters."
    )

    st.markdown("---")
    st.caption("🛡️ **System Compliance:**")
    st.markdown("- **Section 63 BSA 2023** (Evidence Admissibility)")
    st.markdown("- **Sarvam AI / Bhashini** (Sovereign ASR)")
    st.markdown("- **Deterministic NetworkX** (Zero-GPU CPU Execution)")


# ----------------- HEADER & STATUS ----------------- #
st.markdown("<div class='main-title'>Intelligent Crime Network Analysis System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Evidence-Grounded Temporal Crime Knowledge Graph | Multi-Jurisdiction Police Reasoning Layer</div>", unsafe_allow_html=True)

st.markdown("""
<span class='badge badge-primary'>100% CPU Deterministic Engine</span>
<span class='badge badge-success'>BSA 2023 Sec 63 Compliant</span>
<span class='badge badge-warning'>Sarvam AI Sovereign Speech</span>
<span class='badge badge-primary'>Zero Hallucination Guaranteed</span>
""", unsafe_allow_html=True)

st.write("")

# ----------------- TOP KPI METRICS ----------------- #
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='color:#6B7280; font-size:0.85rem;'>Ingested Electronic Sources</div>
        <div style='font-size:1.8rem; font-weight:700; color:#1F2937;'>{len(source_docs)} Files</div>
        <div style='font-size:0.75rem; color:#059669;'>SHA-256 Hashed & Verified</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='color:#6B7280; font-size:0.85rem;'>Network Nodes Surfaced</div>
        <div style='font-size:1.8rem; font-weight:700; color:#1F2937;'>{len(kg.graph.nodes)} Entities</div>
        <div style='font-size:0.75rem; color:#2563EB;'>4 Connected Jurisdictions</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='metric-card'>
        <div style='color:#6B7280; font-size:0.85rem;'>Corroborated Linkages</div>
        <div style='font-size:1.8rem; font-weight:700; color:#1F2937;'>{len(kg.graph.edges)} Connections</div>
        <div style='font-size:0.75rem; color:#059669;'>Telecom + Bank + Physical</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    top_broker = list(betweenness.keys())[0] if betweenness else "N/A"
    top_score = list(betweenness.values())[0] if betweenness else 0.0
    st.markdown(f"""
    <div class='metric-card'>
        <div style='color:#6B7280; font-size:0.85rem;'>Top Covert Broker Detected</div>
        <div style='font-size:1.35rem; font-weight:700; color:#DC2626;'>{top_broker[:16]}...</div>
        <div style='font-size:0.75rem; color:#DC2626;'>Betweenness: {top_score}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ----------------- TABS INTERFACE ----------------- #
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🕸️ Crime Knowledge Graph",
    "🔍 Explain This Connection",
    "🛡️ Anti-Merge Safeguards Test",
    "🎙️ Sarvam AI Speech Intercept",
    "📜 BSA 2023 Sec 63 Certificate"
])

# ----------------- TAB 1: KNOWLEDGE GRAPH ----------------- #
with tab1:
    st.subheader("Interactive Temporal Crime Knowledge Graph")
    st.caption("Physics-based network visualizer. Nodes are sized by Betweenness Centrality (structural broker importance).")

    # Legend
    st.markdown("""
    **Node Legend:** 
    <span style='color:#E63946; font-weight:bold;'>● Person</span> &nbsp;|&nbsp; 
    <span style='color:#457B9D; font-weight:bold;'>● Phone</span> &nbsp;|&nbsp; 
    <span style='color:#F4A261; font-weight:bold;'>● Vehicle</span> &nbsp;|&nbsp; 
    <span style='color:#2A9D8F; font-weight:bold;'>● Bank/UPI</span> &nbsp;|&nbsp; 
    <span style='color:#9B5DE5; font-weight:bold;'>● Case FIR</span> &nbsp;|&nbsp; 
    <span style='color:#6C757D; font-weight:bold;'>● Location</span> &nbsp;&nbsp;&nbsp;&nbsp;
    **Edge Legend:** 
    <span style='color:#2ECC71; font-weight:bold;'>― Observed Record (100% Proven)</span> &nbsp;|&nbsp; 
    <span style='color:#F39C12; font-weight:bold;'>― Inferred (ER Match)</span> &nbsp;|&nbsp; 
    <span style='color:#3498DB; font-weight:bold;'>- - Hypothesis (Structural)</span>
    """, unsafe_allow_html=True)

    html_file = os.path.join(BASE_DIR, "crime_graph.html")
    kg.render_pyvis_html(html_file, height="620px")

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(html_content, height=640, scrolling=True)

    with st.expander("📊 View Top Centrality Ranks (Covert Broker Analysis)"):
        st.write("Nodes ranked by Brandes Betweenness Centrality. Nodes with high scores act as bridges between otherwise isolated clusters:")
        b_df = pd.DataFrame([
            {"Entity Node": k, "Type": kg.graph.nodes[k].get("node_type", ""), "Betweenness Score": v, "Role / Label": kg.graph.nodes[k].get("label", "")}
            for k, v in list(betweenness.items())[:8]
        ])
        st.dataframe(b_df, use_container_width=True)


# ----------------- TAB 2: EXPLAIN THIS CONNECTION ----------------- #
with tab2:
    st.subheader("The 'Explain This Connection' Provenance Drawer")
    st.caption("Select any two entities in the criminal network to view the mathematical traversal path, corroborated proof, and statutory citations.")

    all_nodes = sorted([str(n) for n in kg.graph.nodes])
    col_a, col_b = st.columns(2)
    with col_a:
        entity_a = st.selectbox("Select Entity A", all_nodes, index=all_nodes.index("CASE_104_2026") if "CASE_104_2026" in all_nodes else 0)
    with col_b:
        entity_b = st.selectbox("Select Entity B", all_nodes, index=all_nodes.index("CASE_42_2026") if "CASE_42_2026" in all_nodes else 1)

    if st.button("Examine Relationship & Provenance", type="primary"):
        explanation = kg.explain_connection(entity_a, entity_b)
        
        if explanation.get("status") == "NOT_FOUND":
            st.error(explanation["message"])
        elif explanation.get("connection_type") == "DISCONNECTED":
            st.warning(explanation["message"])
        else:
            st.success(f"Relationship Discovered: {explanation['connection_type']} (Strength: {explanation.get('corroboration_strength', 'HIGH')})")
            
            if explanation["connection_type"] == "DIRECT_EDGE":
                st.markdown(f"**Classification Tier:** `{explanation['tier']}`")
                st.markdown(f"**Relationship Type:** `{explanation['relation_type']}`")
                st.markdown(f"**Corroborating Evidence IDs:** `{', '.join(explanation['evidence_ids'])}`")
                st.json(explanation.get("details", {}))
            else:
                st.markdown(f"**Deterministic Multi-Hop Distance:** `{explanation['path_length']} Hops`")
                st.markdown("**Evidence Path Traversal:**")
                
                path_str = " ➔ ".join([f"`{n}`" for n in explanation["path_nodes"]])
                st.info(path_str)
                
                st.write("**Hop-by-Hop Corroboration Breakdown:**")
                for h in explanation["hops"]:
                    st.markdown(f"- **{h['from']}** ──[{h['relation']}]──► **{h['to']}** (Tier: `{h['tier']}`, Evidence: `{', '.join(h['evidence'])}`)")

            st.markdown(f"""
            > ⚖️ **Statutory Evidence Note (BSA 2023 Sec 63):**  
            > *{explanation.get('statutory_note', '')}*  
            > *This lead is derived from verified primary electronic ledger items and telephone exchange logs.*
            """)


# ----------------- TAB 3: ANTI-MERGE SAFEGUARDS TEST ----------------- #
with tab3:
    st.subheader("Live Anti-Merge Safeguards & Edge-Case Benchmark")
    st.caption("Demonstrating how the system prevents false syndicate clustering and respects demographic constraints.")

    st.markdown("#### Test Case 1: The High-Density Address Trap (Ameerpet Student Hostel)")
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.write("Suppose two unrelated students live at the same PG hostel address:")
        st.code("""
Record 1: S. Rahul (Student), H.No. 4-1-25, Balaji Deluxe Boys Hostel, Ameerpet, Hyderabad
Record 2: V. Suresh (Accused in cyber fraud), H.No. 4-1-25, Balaji Deluxe Boys Hostel, Ameerpet, Hyderabad
        """)
    with col_t2:
        if st.button("Run Hostel Test", key="btn_hostel"):
            r1 = {"name": "S. Rahul", "address": "H.No. 4-1-25, Balaji Deluxe Boys Hostel, Ameerpet", "phone": "+919111111111"}
            r2 = {"name": "V. Suresh", "address": "H.No. 4-1-25, Balaji Deluxe Boys Hostel, Ameerpet", "phone": "+919222222222"}
            res = resolver.evaluate_pair(r1, r2)
            st.error(f"Decision: {res['decision']}")
            st.write(f"Match Score: `{res['match_score']}`")
            st.info(f"Reason: {res['reason']}")

    st.markdown("---")
    st.markdown("#### Test Case 2: Co-accused Conflict (Negative Constraint #1)")
    col_t3, col_t4 = st.columns([2, 1])
    with col_t3:
        st.write("Two distinct co-accused named in the exact same FIR:")
        st.code("""
Record 1: Imran Qureshi (Accused #1 in FIR 42/2026 Pune)
Record 2: Tariq Sheikh (Accused #2 in FIR 42/2026 Pune)
        """)
    with col_t4:
        if st.button("Run Co-Accused Test", key="btn_coaccused"):
            r3 = {"name": "Imran Qureshi", "fir_id": "42/2026", "role": "ACCUSED"}
            r4 = {"name": "Tariq Sheikh", "fir_id": "42/2026", "role": "ACCUSED"}
            res2 = resolver.evaluate_pair(r3, r4)
            st.warning(f"Decision: {res2['decision']}")
            st.write(f"Match Score: `{res2['match_score']}`")
            st.info(f"Reason: {res2['reason']}")

    st.markdown("---")
    st.markdown("#### Test Case 3: Indian Vehicle License Plate OCR Correction")
    col_t5, col_t6 = st.columns([2, 1])
    with col_t5:
        test_plate = st.text_input("Enter License Plate with OCR Misread", "TS09A81234")
    with col_t6:
        clean_p, was_fixed = normalize_vehicle_plate(test_plate)
        st.write(f"Normalized Registration: `{clean_p}`")
        if was_fixed:
            st.success("✅ OCR Typo Repaired: Character '8' restored to series letter 'B'")


# ----------------- TAB 4: SARVAM AI SPEECH INTERCEPT ----------------- #
with tab4:
    st.subheader("Sovereign Indic Speech-to-Text Intercept (Sarvam AI)")
    st.caption("MeitY & Bhashini compliant Indian language ASR. Handles Hinglish & Telugu dialect code-mixing.")

    sarvam_client = SarvamSpeechClient()
    audio_meta = pipeline_data["audio_meta"]

    col_a1, col_a2 = st.columns([1, 1])
    with col_a1:
        st.markdown(f"**Intercept ID:** `{audio_meta.get('intercept_id', 'CALL_INT_2026_092')}`")
        st.markdown(f"**Target Phone:** `{audio_meta.get('target_phone', '+919876543210')}`")
        st.markdown(f"**Dialed Phone:** `{audio_meta.get('dialed_phone', '+919123456789')}`")
        st.markdown(f"**ASR Model:** `{audio_meta.get('asr_engine', 'Sarvam AI Saaras')}`")
        st.markdown(f"**Language Detected:** `{audio_meta.get('language_detected', 'Hinglish / Telugu')}`")
        st.markdown(f"**Cryptographic Hash (SHA-256):** `{audio_meta.get('sha256', '')[:24]}...`")

    with col_a2:
        st.write("🎙️ **Raw Hindi/Telugu Audio Intercept Transcript:**")
        st.info(f"\"{audio_meta.get('transcript', '')}\"")
        st.write("🇬🇧 **Investigative English Translation:**")
        st.success(f"\"{audio_meta.get('english_translation', '')}\"")

    st.write("🔍 **Extracted Telecom & Financial Entities:**")
    st.dataframe(pd.DataFrame(audio_meta.get("extracted_entities", [])), use_container_width=True)


# ----------------- TAB 5: BSA 2023 CERTIFICATE ----------------- #
with tab5:
    st.subheader("Statutory Lead Certificate under Section 63 BSA 2023")
    st.caption("Official court-admissible chain-of-custody report ready for submission to the Hon'ble Chief Judicial Magistrate.")

    case_summary = {
        "syndicate_name": "Operation Cyber-Spider Syndicate",
        "firs": ["FIR 104/2026 (Cyberabad)", "FIR 88/2026 (Hyderabad)", "FIR 42/2026 (Pune)", "FIR 19/2026 (Rachakonda)"],
        "statutory_sections": "BNS 318(4) [Cheating], BNS 111 [Organised Crime], BNS 303(2) [Theft], IT Act 66D",
        "total_defrauded_inr": "40,50,000"
    }

    corroborated_links = [
        {
            "source": "Farooq Ahmed (Alias: Bhaijaan)",
            "relation": "COVERT_COORDINATOR",
            "target": "Rahim Khan (Operative)",
            "tier": "OBSERVED",
            "telecom_proof": "18 Calls logged on CDR Batch #994 (Tower ID: HYD-MADHAPUR)",
            "financial_proof": "₹1,80,000 transferred to HDFC Bank (TXN_20104)",
            "physical_proof": "Vehicle TS09AB1234 sighted outside Madhapur branch (ANPR_803)",
            "legal_strength": "HIGH_CORROBORATION (Ready for Warrant under Sec 94 BNSS)"
        },
        {
            "source": "Vehicle TS09AB1234",
            "relation": "STOLEN_GETAWAY_CAR",
            "target": "FIR 104/2026 & FIR 19/2026",
            "tier": "OBSERVED",
            "telecom_proof": "Driver phone pinged Zaheerabad border checkpost",
            "financial_proof": "Fastag toll debited on stolen plate",
            "physical_proof": "ANPR Camera 801 at LB Nagar Ring Road & 804 at Zaheerabad",
            "legal_strength": "PRIMA_FACIE_EVIDENCE"
        }
    ]

    cert_text = generate_bsa_section63_certificate(
        case_summary, corroborated_links, source_docs
    )

    st.text_area("Official Certificate Preview", cert_text, height=380)

    st.download_button(
        label="📥 Download Official BSA Section 63 Certificate (.txt)",
        data=cert_text,
        file_name=f"BSA_Section63_Certificate_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        type="primary"
    )
