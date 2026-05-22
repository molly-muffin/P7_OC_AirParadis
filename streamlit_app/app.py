"""OC-compliant Streamlit interface for testing the sentiment API."""

import os
from datetime import datetime

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Air Paradis - Test Sentiment API", page_icon="✈️", layout="centered")

st.title("Air Paradis — Test de l'API Sentiment")
st.caption("Interface locale pour tester les prédictions et remonter les erreurs à Application Insights")

api_url = st.sidebar.text_input("URL de l'API", value=API_BASE_URL)

tweet = st.text_area("Saisissez un tweet", height=120, placeholder="Ex: Air Paradis flight was amazing!")

if st.button("Analyser le sentiment", type="primary"):
    if not tweet.strip():
        st.warning("Entrez un tweet.")
    else:
        try:
            resp = requests.post(
                f"{api_url.rstrip('/')}/predict",
                json={"text": tweet},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["last_prediction"] = data
                st.session_state["last_tweet"] = tweet
                sentiment = data["sentiment"]
                emoji = "😊" if sentiment == "positive" else "😞"
                st.success(f"{emoji} Sentiment prédit : **{sentiment.upper()}**")
                st.metric("Confiance", f"{data['confidence'] * 100:.1f}%")
                st.caption(f"Modèle : {data.get('model', 'N/A')} | ID : {data.get('prediction_id', 'N/A')}")
            else:
                st.error(f"Erreur API {resp.status_code}: {resp.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Impossible de joindre l'API à {api_url}. Lancez l'API avec : uvicorn src.api.main:app --reload")

if "last_prediction" in st.session_state:
    st.divider()
    st.subheader("La prédiction est-elle correcte ?")
    col1, col2 = st.columns(2)
    pred = st.session_state["last_prediction"]
    tweet_text = st.session_state["last_tweet"]

    with col1:
        if st.button("✅ Oui, correcte"):
            st.success("Merci pour votre retour !")

    with col2:
        if st.button("❌ Non, incorrecte"):
            actual = st.selectbox(
                "Quel est le vrai sentiment ?",
                ["negative", "positive"],
                index=0 if pred["sentiment"] == "positive" else 1,
            )
            try:
                fb = requests.post(
                    f"{api_url.rstrip('/')}/feedback",
                    json={
                        "text": tweet_text,
                        "predicted_sentiment": pred["sentiment"],
                        "actual_sentiment": actual,
                        "prediction_id": pred.get("prediction_id"),
                    },
                    timeout=10,
                )
                if fb.status_code == 200:
                    result = fb.json()
                    st.warning("Retour enregistré et trace envoyée à Application Insights.")
                    if result.get("alert_triggered"):
                        st.error(
                            "⚠️ ALERTE : 3 tweets mal prédits en 5 minutes — "
                            "notification déclenchée (SMS/email simulé via App Insights)."
                        )
                else:
                    st.error(f"Erreur feedback: {fb.text}")
            except requests.exceptions.ConnectionError:
                st.error("API inaccessible pour le feedback.")

st.sidebar.divider()
st.sidebar.markdown("### Santé API")
if st.sidebar.button("Vérifier /health"):
    try:
        h = requests.get(f"{api_url.rstrip('/')}/health", timeout=5)
        st.sidebar.json(h.json())
    except Exception as exc:
        st.sidebar.error(str(exc))

st.sidebar.caption(f"Dernière mise à jour : {datetime.now().strftime('%H:%M:%S')}")
