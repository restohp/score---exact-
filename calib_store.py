"""Calculs d'auto-correction mis en cache (partagés par les pages Cartes et Auto-correction)."""
import streamlit as st

import calib_core as cb


@st.cache_data(ttl=43200, show_spinner=False)
def predictions(code):
    """Prédictions « dans le passé » d'un championnat (1 à 3 minutes la première fois)."""
    return cb.oos_predictions(cb.load_data(code, n_seasons=5))


@st.cache_data(ttl=43200, show_spinner=False)
def calibration(code):
    """Correction complète : globale + par équipe (si elle aide) + part de buts en 1re mi-temps."""
    return cb.fit_full(predictions(code))
