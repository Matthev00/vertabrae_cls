import streamlit as st
import pandas as pd
import requests
from pathlib import Path
from typing import List, Dict
import plotly.graph_objects as go
from src.config import DICOM_DATA_DIR


API_URL = "http://localhost:8000/predict_fake"
VERTEBRA_ORDER = [f"C{i}" for i in range(1, 8)] + [f"Th{i}" for i in range(1, 13)] + [f"L{i}" for i in range(1, 6)]


def get_dicom_dirs() -> List[str]:
    """
    Get a list of available DICOM directories.

    Returns:
        List[str]: Sorted list of directory names.
    """
    return sorted([p.name for p in DICOM_DATA_DIR.iterdir() if p.is_dir()])


def fetch_predictions(patient_id: str, k: int) -> Dict:
    """
    Fetch predictions from the API.

    Args:
        patient_id (str): Selected patient ID.
        k (int): Number of top-k classes.

    Returns:
        Dict: JSON response from the API.
    """
    params = {"patient_id": patient_id, "k": k}
    response = requests.get(API_URL, params=params)
    return response.json()


def process_predictions(data: Dict, k: int) -> pd.DataFrame:
    """
    Process predictions into a DataFrame.

    Args:
        data (Dict): API response data.
        k (int): Number of top-k classes.

    Returns:
        pd.DataFrame: Processed predictions DataFrame.
    """
    rows = []
    for vertebra in VERTEBRA_ORDER:
        row = {"Vertebra": vertebra}
        preds = next((entry["topk"] for entry in data["predictions"] if entry["vertebra"] == vertebra), None)
        if preds:
            for i in range(k):
                if i < len(preds):
                    cls, prob = preds[i]
                    row[f"Top-{i+1} Class"] = cls
                    row[f"Top-{i+1} Prob"] = float(prob)
        else:
            for i in range(k):
                row[f"Top-{i+1} Class"] = ""
                row[f"Top-{i+1} Prob"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def process_heatmap_data(data: Dict) -> pd.DataFrame:
    """
    Process data for heatmap visualization.

    Args:
        data (Dict): API response data.

    Returns:
        pd.DataFrame: Heatmap data DataFrame.
    """
    heatmap_data = []
    for vertebra in VERTEBRA_ORDER:
        preds = next((entry["topk"] for entry in data["predictions"] if entry["vertebra"] == vertebra), None)
        if preds:
            cls0, prob0 = preds[0]
            heatmap_data.append({"Vertebra": vertebra, "Class": cls0, "Probability": prob0})
        else:
            heatmap_data.append({"Vertebra": vertebra, "Class": "", "Probability": 0.0})
    return pd.DataFrame(heatmap_data)


def style_predictions(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Apply styling to the predictions DataFrame.

    Args:
        df (pd.DataFrame): Predictions DataFrame.

    Returns:
        pd.io.formats.style.Styler: Styled DataFrame.
    """
    def color_row(row):
        cls = row.get("Top-1 Class")
        prob = row.get("Top-1 Prob")
        if pd.isna(prob) or cls is None:
            return [""] * len(row)
        prob = float(prob)
        if cls == "H":
            return [f"background-color: rgba(0,255,0,{0.3 * prob}); color: black"] * len(row)
        return [f"background-color: rgba(255,0,0,{0.3 * prob}); color: white"] * len(row)

    return df.style.apply(color_row, axis=1)


def create_heatmap(heatmap_df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap visualization.

    Args:
        heatmap_df (pd.DataFrame): Heatmap data DataFrame.

    Returns:
        go.Figure: Plotly heatmap figure.
    """
    healthy_bars = heatmap_df[heatmap_df["Class"] == "H"].copy()
    healthy_bars["Value"] = -1 * healthy_bars["Probability"]

    injury_bars = heatmap_df[heatmap_df["Class"] != "H"].copy()
    injury_bars["Value"] = injury_bars["Probability"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=healthy_bars["Value"],
        y=healthy_bars["Vertebra"],
        orientation='h',
        name="Zdrowy",
        marker_color="green",
        hovertemplate="Kręg: %{y}<br>H: %{x:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=injury_bars["Value"],
        y=injury_bars["Vertebra"],
        orientation='h',
        name="Uraz",
        marker_color="crimson",
        hovertemplate="Kręg: %{y}<br>Uraz: %{x:.2f}<extra></extra>",
    ))

    fig.update_layout(
        barmode='relative',
        xaxis_title="Prawdopodobieństwo",
        yaxis=dict(categoryorder="array", categoryarray=VERTEBRA_ORDER[::-1]),
        height=800,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
    )

    return fig


def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(layout="wide")

    dicom_dirs = get_dicom_dirs()
    selected_patient = st.sidebar.selectbox("Pacjent (folder)", dicom_dirs)
    k = st.sidebar.slider("Liczba top-k klas", min_value=1, max_value=5, value=2)

    data = fetch_predictions(selected_patient, k)
    df = process_predictions(data, k)
    heatmap_df = process_heatmap_data(data)

    styled_df = style_predictions(df)

    st.title(f"Wyniki klasyfikacji – {selected_patient}")
    col1, col2 = st.columns([3, 1], gap="large")

    with col1:
        st.subheader("Tabela predykcji")
        st.dataframe(styled_df, use_container_width=True, height=800)

    with col2:
        st.subheader("Prawdopodobieństwo (zdrowy vs uraz)")
        fig = create_heatmap(heatmap_df)
        st.plotly_chart(fig, use_container_width=True)

    unfound = data.get("unfound_vertebrae", [])
    if unfound:
        st.warning("Nie wykryto kręgów: " + ", ".join(sorted(unfound)))


if __name__ == "__main__":
    main()