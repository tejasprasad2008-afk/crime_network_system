# Intelligent Crime Network Analysis System (ICNAS)
### A Deterministic Knowledge Graph & Forensic Reasoning Layer for Inter-Jurisdictional Law Enforcement

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Compliance](https://img.shields.io/badge/Legal%20Standard-Section%2063%20BSA%202023-crimson.svg)](#legal-provenance-engine)
[![Engine](https://img.shields.io/badge/Architecture-100%25%20CPU%20Deterministic-blueviolet.svg)](#system-architecture)

---

## 1. Executive Summary & Field Proposal

India's 17,000+ police stations are digitized under **CCTNS** (Crime and Criminal Tracking Network & Systems) and interoperable through **ICJS** (Inter-operable Criminal Justice System). However, investigators frequently encounter severe cognitive bottlenecks when trying to connect fragmented electronic records across state jurisdictions.

When an organized cyber-fraud or extortion syndicate strikes:
- The **FIR** is lodged in Cyberabad.
- The **money mule ATM withdrawal** happens in Madhapur.
- The **stolen getaway car** passes an ANPR toll on NH-65.
- The **hawala coordinator** operates out of Pune without ever directly contacting the victim.

Today, current police workflows rely on **siloed relational search portals** (Vahan for cars, NAFIS for fingerprints, standalone CDR software for phone dumps). If an investigator does not already suspect a specific phone number or vehicle, the cross-state link remains buried for months.

**ICNAS** is an open, field-deployable reference prototype that provides the **automated reasoning and multi-modal inference layer** above CCTNS/ICJS:
- **100% Deterministic & Zero-GPU**: Runs directly on standard 8GB RAM government laptops using pure CPU graph algorithms—guaranteeing zero probabilistic hallucinations in criminal evidence chains.
- **Section 63 Bharatiya Sakshya Adhiniyam (BSA), 2023 Compliant**: Automatically computes SHA-256 cryptographic chain-of-custody trees and generates statutory lead certificates for magistrate search warrants under Section 94 BNSS, 2023.
- **Hard Anti-Merge Safeguards & Responsible AI**: Implements an Address Entropy Benchmark that prevents false arrests by blocking entity merges on high-occupancy addresses (e.g., student hostels or PGs).
- **Domain-Specific Resolvers**: Includes an Indian RTO positional plate normalizer and Indic phonetic matching to resolve dialect spelling variants.
- **Sovereign Speech Intercept**: Integrates sovereign Indian speech-to-text (Sarvam AI / MeitY Bhashini standards) for Hinglish and regional dialect call intercepts.

---

## 2. System Architecture

```
                                  [ MULTI-MODAL EVIDENCE INGESTION ]
                                                 │
        ┌───────────────────────┬────────────────┼───────────────────────┬───────────────────────┐
        ▼                       ▼                ▼                       ▼                       ▼
    [ FIR Text ]          [ CDR Dumps ]    [ UPI / Bank ]          [ ANPR Toll ]          [ Audio Intercept ]
   (Semi-Structured)       (CSV Records)   (Ledger CSV)            (CCTV Plates)          (Hinglish / Telugu)
        │                       │                │                       │                       │
        └───────────────────────┴────────────────┼───────────────────────┴───────────────────────┘
                                                 ▼
                              [ SHA-256 Chain of Custody Ingestion ]
                                                 ▼
                       [ Deterministic Entity Resolution & Disambiguation ]
                         ├── Indian RTO Vehicle Plate OCR Normalizer
                         ├── Indic Phonetic Soundex + Jaro-Winkler Matching
                         └── Hard Anti-Merge Safeguards (Address Entropy Gating)
                                                 ▼
                        [ NetworkX CPU Knowledge Graph Analytics Engine ]
                         ├── Multi-Source Classified Edges (Observed / Inferred)
                         ├── Brandes Betweenness Centrality (O(V·E) Kingpin Detection)
                         └── Multi-Hop Shortest Path Provenance Tracer
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             [ Interactive Visualizer ]                   [ Statutory BSA Certificate ]
         (Web Cockpit & Streamlit UI)                 (Court-Admissible Sec 63 BSA 2023)
```

---

## 3. Core Technical Innovations

### A. Algorithmic Anti-Merge Safeguards (Responsible AI)
In criminal law, a false positive ruins innocent lives. Naive graph clustering algorithms often link people solely based on shared co-location. In high-density rental hubs (such as Ameerpet in Hyderabad or Mukherjee Nagar in Delhi), 100+ students share a single postal address. 
- ICNAS incorporates an **Address Entropy Rule**: when an address matches known multi-tenant keywords or registers with high occupant count, the address similarity weight is slashed from $+0.25$ down to an epsilon $+0.02$.
- Distinct co-accused in the same FIR are subject to hard-coded anti-merge constraints, guaranteeing distinct identities are never collapsed into one node.

### B. Brandes Betweenness Centrality for Kingpin Unmasking
Masterminds of cyber-syndicates intentionally avoid direct contact with victims. Standard CDR analyzers that look only at degree connectivity fail to highlight them.
- ICNAS executes **Brandes Betweenness Centrality** ($O(V \cdot E)$):
  $$C_B(v) = \sum_{s \ne v \ne t} \frac{\sigma_{st}(v)}{\sigma_{st}}$$
- Nodes mediating information and resource flows across disparate cases (such as hawala brokers) glow with the highest centrality scores, unmasking shadow operators.

### C. Indian RTO Positional Plate Normalizer
Highway ANPR cameras frequently confuse alphanumeric characters (e.g. reading `8` instead of `B`). Using standard Indian RTO registration syntax (`[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}`), the positional normalizer repairs optical glitches (e.g., `TS09A81234` $\to$ `TS09AB1234`) deterministically before graph ingestion.

### D. Statutory Legal Admissibility (Section 63 BSA 2023)
Effective July 1, 2024, the Indian Evidence Act 1872 was repealed by the **Bharatiya Sakshya Adhiniyam, 2023**. Section 65B was replaced by **Section 63 BSA**.
- ICNAS computes cryptographic SHA-256 hashes of all ingested files upon arrival.
- It automatically generates court-ready certificates detailing input hashes, deterministic provenance paths, and statutory declarations required by magistrates under Section 94 BNSS, 2023.

---

## 4. Repository Structure

```
crime_network_system/
├── app.py                      # Streamlit Law Enforcement Cockpit
├── run_icnas.bat               # 1-Click Launch Script
├── requirements.txt            # Python Dependencies
├── data/
│   ├── known_hostels_pms.json  # Multi-tenant address entropy benchmark
│   └── sample_cases/           # Synthetic "Operation Cyber-Spider" dataset
│       ├── fir_104_cyberabad.txt   # Digital Arrest Case (INR 28.5L)
│       ├── fir_88_hyderabad.txt    # Courier Customs Case (INR 12L)
│       ├── fir_42_pune.txt         # Hawala Raid (Cash INR 64.5L seized)
│       ├── fir_19_rachakonda.txt   # Stolen White Swift TS09AB1234
│       ├── cdr_dump_feb2026.csv    # 12 Call records linking numbers & towers
│       ├── bank_upi_transactions.csv# Bank accounts & UPI fund flow
│       ├── anpr_toll_sightings.csv # Highway toll & CCTV plate sightings
│       └── audio_transcript_intercept.json # Wiretap transcript
├── modules/
│   ├── __init__.py
│   ├── ingestion.py            # SHA-256 hasher & deterministic entity extractor
│   ├── entity_resolution.py    # Indic Soundex, Jaro-Winkler, Plate normalizer, Anti-merge
│   ├── graph_engine.py         # NetworkX graph builder, Betweenness Centrality, PyVis renderer
│   ├── sarvam_speech.py        # Sovereign Indic ASR client with air-gapped fallback
│   └── bsa_report.py           # Statutory Section 63 BSA legal certificate generator
└── web/
    ├── index.html              # Standalone Cytoscape.js Dark Tactical Web Dashboard
    ├── package.json
    └── assets/                 # Case narrative multimedia evidence
```

---

## 5. Quickstart & Verification

### Prerequisites
- Python 3.10+
- 8GB RAM minimum (Zero GPU required)

### Installation
```bash
# Clone the repository
git clone https://github.com/tejasprasad2008-afk/crime_network_system.git
cd crime_network_system

# Install dependencies
pip install -r requirements.txt
```

### Running the Streamlit Forensic Cockpit
```bash
streamlit run app.py
```
*Access at: `http://localhost:8501`*

### Running the Standalone Web Dashboard
```bash
cd web
python3 -m http.server 3000
```
*Access at: `http://localhost:3000`*

---

## 6. Verification & Evaluation Walkthrough

1. **Chain of Custody**: View the top telemetry bar confirming 5 ingested electronic sources with verified SHA-256 cryptographic hashes.
2. **Knowledge Graph Visualizer (Tab 1)**: Observe how 4 cross-state FIRs (Cyberabad, Hyderabad, Rachakonda, Pune) assemble into a connected syndicate. Notice `Farooq Ahmed (Bhaijaan)` highlighted with the highest betweenness centrality score.
3. **Multi-Hop Provenance (Tab 2)**: Select `CASE_104_2026` (Cyberabad) and `CASE_42_2026` (Pune). Click **Examine Provenance** to view the 4-hop audit trail: `Victim → IndusInd Mule → Madhapur ATM → White Swift TS09AB1234 → Pune Hawala`.
4. **Anti-Merge Guardrails (Tab 3)**: Execute the **Hostel Test** and **Co-Accused Test** to verify that address entropy down-weighting blocks false merges for shared residential complexes.
5. **Sovereign Speech (Tab 4)**: Inspect the transcribed code-mixed Hinglish wiretap intercept.
6. **Statutory BSA Certificate (Tab 5)**: Generate and export the Section 63 BSA 2023 certificate ready for magistrate review.

---

## 7. Production Viability & Future Scope

| Feature | Prototype Status | Production Roadmap |
| :--- | :--- | :--- |
| **Graph Backend** | In-memory NetworkX ($<25,000$ entities) | Distributed Neo4j / Memgraph cluster with Redis caching |
| **CCTNS Integration** | Batch file ingestion (FIR text / CSV) | Kafka message bus ingesting live ICJS XML/JSON event streams |
| **Entity Resolution** | Indic Phonetic + Jaro-Winkler + Entropy | Federated block-level linkage with state-specific soundex tables |
| **Deployment** | Local / Docker single container | State Data Centre (SDC) on-premise air-gapped deployment |

---

## 8. License & Disclaimer

Distributed under the **MIT License**.

> **Note on Data:** All names, telephone numbers, bank accounts, vehicle registrations, and case scenarios included in this repository are **entirely synthetic and generated for demonstration and verification purposes only**. They do not represent real persons, active investigations, or actual law enforcement records.
