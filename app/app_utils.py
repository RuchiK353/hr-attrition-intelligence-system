"""Shared, cached loaders for the Streamlit app."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, features, predict, preprocessing, train  # noqa: E402


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    raw = preprocessing.load_raw_data()
    clean = preprocessing.clean_data(raw)
    fe = features.engineer_features(clean)
    return fe


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return predict.load_model_bundle()


@st.cache_data(show_spinner=False)
def get_results_table() -> pd.DataFrame:
    if config.RESULTS_PATH.exists():
        return pd.read_csv(config.RESULTS_PATH)
    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def get_explainer_cache():
    return {}
