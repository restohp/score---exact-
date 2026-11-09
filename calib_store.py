"""Stockage en cache des prédictions « dans le passé » et de la correction apprise."""
import streamlit as st

import calib_core as cb
import forme_core as fc


@st.cache_data(ttl=43200, show_spinner=False)
def predictions(code):
    """Prédictions refaites dans le passé pour un championnat (recalculées toutes les 12 h)."""
    return cb.oos_predictions(fc.load_stats(code, n_seasons=5))


@st.cache_data(ttl=43200, show_spinner=False)
def calibration(code):
    """Correction apprise sur les erreurs passées du championnat."""
    return cb.fit_calibration(predictions(code))
