import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_URL", "http://api:8000")
API_URL = f"{API_BASE}/predict"
METRICS_PATH = Path(os.getenv("METRICS_PATH", "reports/model_metrics.json"))

FEATURES = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",
    "EPS1", "FS1", "FS2",
    "TS1", "TS2", "TS3", "TS4",
    "VS1", "CE", "CP", "SE",
]

# ── Labels en langage naturel ──────────────────────────────────────────────────

COMPONENT_LABELS = {
    "cooler_condition": {
        3:   ("Défaillance imminente",     "🔴"),
        20:  ("Efficacité réduite",         "🟡"),
        100: ("Nominal",                    "🟢"),
    },
    "valve_condition": {
        73:  ("Défaillance imminente",      "🔴"),
        80:  ("Décalage sévère",            "🟠"),
        90:  ("Léger décalage",             "🟡"),
        100: ("Comportement optimal",       "🟢"),
    },
    "pump_leakage": {
        0:   ("Aucune fuite",               "🟢"),
        1:   ("Fuite faible",               "🟡"),
        2:   ("Fuite sévère",               "🔴"),
    },
    "accumulator_pressure": {
        90:  ("Défaillance imminente",      "🔴"),
        100: ("Pression fortement réduite", "🟠"),
        115: ("Pression légèrement réduite","🟡"),
        130: ("Pression optimale",          "🟢"),
    },
}

COMPONENT_NAMES = {
    "cooler_condition":      "Refroidisseur",
    "valve_condition":       "Valve",
    "pump_leakage":          "Pompe",
    "accumulator_pressure":  "Accumulateur",
}

# ── Presets de cas de test ─────────────────────────────────────────────────────

PRESETS = {
    "Système nominal": {
        "PS1": 161.011, "PS2": 109.635, "PS3": 1.998, "PS4": 10.064,
        "PS5": 9.836, "PS6": 9.721, "EPS1": 2539.066,
        "FS1": 6.701, "FS2": 10.166,
        "TS1": 36.255, "TS2": 41.861, "TS3": 39.138, "TS4": 31.179,
        "VS1": 0.551, "CE": 47.824, "CP": 2.180, "SE": 59.163,
    },
    "Refroidisseur en panne": {
        "PS1": 173.092, "PS2": 122.552, "PS3": 1.087, "PS4": 0.0001,
        "PS5": 8.450, "PS6": 8.401, "EPS1": 2630.216,
        "FS1": 3.231, "FS2": 8.994,
        "TS1": 56.279, "TS2": 60.317, "TS3": 57.835, "TS4": 51.881,
        "VS1": 0.738, "CE": 18.654, "CP": 1.444, "SE": 29.275,
    },
    "Valve défaillante": {
        "PS1": 161.271, "PS2": 109.049, "PS3": 1.984, "PS4": 10.099,
        "PS5": 9.864, "PS6": 9.748, "EPS1": 2541.645,
        "FS1": 6.667, "FS2": 10.173,
        "TS1": 36.034, "TS2": 41.601, "TS3": 38.896, "TS4": 30.979,
        "VS1": 0.545, "CE": 47.703, "CP": 2.171, "SE": 58.161,
    },
    "Fuite sévère de pompe": {
        "PS1": 160.454, "PS2": 109.251, "PS3": 1.927, "PS4": 10.179,
        "PS5": 9.947, "PS6": 9.828, "EPS1": 2581.368,
        "FS1": 6.436, "FS2": 10.209,
        "TS1": 35.721, "TS2": 41.258, "TS3": 38.616, "TS4": 30.729,
        "VS1": 0.537, "CE": 47.247, "CP": 2.169, "SE": 55.714,
    },
    "Système dégradé": {
        "PS1": 158.469, "PS2": 107.434, "PS3": 1.750, "PS4": 0.0,
        "PS5": 9.170, "PS6": 9.085, "EPS1": 2485.705,
        "FS1": 6.359, "FS2": 9.735,
        "TS1": 44.109, "TS2": 49.043, "TS3": 46.364, "TS4": 39.744,
        "VS1": 0.623, "CE": 27.414, "CP": 1.734, "SE": 56.257,
    },
}

PRESET_DESCRIPTIONS = {
    "Système nominal":       "Tous les composants en état optimal.",
    "Refroidisseur en panne": "Températures élevées, efficacité de refroidissement quasi nulle.",
    "Valve défaillante":     "Valve proche de la défaillance totale, reste du système stable.",
    "Fuite sévère de pompe": "Fuite hydraulique sévère détectée sur la pompe.",
    "Système dégradé":       "Refroidisseur et accumulateur en état dégradé simultanément.",
}

# ── Utilitaires ────────────────────────────────────────────────────────────────

def label_for(component: str, value: int) -> tuple[str, str]:
    mapping = COMPONENT_LABELS.get(component, {})
    return mapping.get(value, (str(value), "⚪"))


def send_prediction(inputs: dict) -> dict | None:
    try:
        response = requests.post(API_URL, json=inputs, timeout=10)
        if response.status_code == 200:
            return response.json()
        st.error(f"Erreur API {response.status_code}: {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre l'API (port 8000).")
    except Exception as exc:
        st.error(f"Erreur inattendue : {exc}")
    return None


def render_prediction_result(preds: dict) -> None:
    cols = st.columns(len(preds))
    for col, (component, raw_value) in zip(cols, preds.items()):
        label, icon = label_for(component, raw_value)
        name = COMPONENT_NAMES.get(component, component)
        with col:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    padding: 1rem;
                    text-align: center;
                    background: #1e1e2e;
                ">
                    <div style="font-size: 2.5rem">{icon}</div>
                    <div style="font-weight: bold; font-size: 1rem; margin: 0.4rem 0">{name}</div>
                    <div style="font-size: 0.85rem; color: #ccc">{label}</div>
                    <div style="font-size: 0.75rem; color: #888; margin-top: 0.3rem">classe {raw_value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── App ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Hydraulic Condition Monitor", layout="wide")
st.title("Hydraulic Condition Monitor")

tab1, tab2 = st.tabs(["Prédiction", "Évaluation du modèle"])

with tab1:
    # ── Presets ────────────────────────────────────────────────────────────────
    st.subheader("Cas de test rapides")
    preset_cols = st.columns(len(PRESETS))
    selected_preset = st.session_state.get("preset_values", PRESETS["Système nominal"])

    for col, (name, values) in zip(preset_cols, PRESETS.items()):
        with col:
            if st.button(name, use_container_width=True):
                st.session_state["preset_values"] = values
                st.session_state["preset_name"] = name
                st.rerun()

    preset_name = st.session_state.get("preset_name", "Système nominal")
    current_values = st.session_state.get("preset_values", PRESETS["Système nominal"])

    st.caption(f"**{preset_name}** — {PRESET_DESCRIPTIONS[preset_name]}")
    st.divider()

    # ── Capteurs ───────────────────────────────────────────────────────────────
    st.subheader("Valeurs des capteurs")

    SENSOR_GROUPS = {
        "Pression (bar)":       ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6"],
        "Puissance / Débit":    ["EPS1", "FS1", "FS2"],
        "Température (°C)":     ["TS1", "TS2", "TS3", "TS4"],
        "Vibration / Divers":   ["VS1", "CE", "CP", "SE"],
    }

    inputs = {}
    for group_name, group_features in SENSOR_GROUPS.items():
        with st.expander(group_name, expanded=True):
            gcols = st.columns(len(group_features))
            for gcol, feat in zip(gcols, group_features):
                with gcol:
                    inputs[feat] = st.number_input(
                        feat,
                        value=float(current_values.get(feat, 0.0)),
                        format="%.4f",
                        key=f"input_{feat}",
                    )

    st.divider()

    if st.button("Lancer la prédiction", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours…"):
            result = send_prediction(inputs)
        if result:
            preds = result.get("predictions", {})
            if preds:
                st.subheader("État des composants")
                render_prediction_result(preds)
            else:
                st.warning("Aucune prédiction retournée par l'API.")

with tab2:
    st.subheader("Évaluation du modèle")

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        for target, values in metrics.items():
            st.markdown(f"### {COMPONENT_NAMES.get(target, target)}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  f"{values['accuracy']:.3f}")
            c2.metric("Precision", f"{values['precision_weighted']:.3f}")
            c3.metric("Recall",    f"{values['recall_weighted']:.3f}")
            c4.metric("F1",        f"{values['f1_weighted']:.3f}")

            st.write("Matrice de confusion")
            cm_df = pd.DataFrame(values["confusion_matrix"])
            st.dataframe(cm_df, use_container_width=False)

            with st.expander("Rapport de classification détaillé"):
                report_df = pd.DataFrame(values["classification_report"]).transpose()
                st.dataframe(report_df, use_container_width=True)

            st.divider()
    else:
        st.warning(
            "Fichier de métriques introuvable. Lancez d'abord le pipeline d'entraînement "
            "pour générer reports/model_metrics.json."
        )
