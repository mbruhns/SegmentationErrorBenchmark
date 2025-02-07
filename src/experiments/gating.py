import numpy as np
from sklearn.mixture import GaussianMixture
import pandas as pd


def gmm_marker(data, marker, n_components=3):
    x = data[marker].values.reshape(-1, 1)
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(x)

    # Sort the components by their means to ensure consistent class labels
    sorted_indices = np.argsort(gmm.means_.flatten())
    gmm.means_ = gmm.means_[sorted_indices]
    gmm.covariances_ = gmm.covariances_[sorted_indices]
    gmm.weights_ = gmm.weights_[sorted_indices]
    gmm.precisions_ = gmm.precisions_[sorted_indices]
    gmm.precisions_cholesky_ = gmm.precisions_cholesky_[sorted_indices]

    # Predict and adjust based on conditions
    prediction = gmm.predict(x) // (n_components - 1)
    prediction[x.ravel() > gmm.means_.max()] = 1
    proba_ = gmm.predict_proba(x)

    prediction = prediction.astype(int)
    proba_ = np.vstack([proba_[:, :2].sum(axis=1), proba_[:, -1]]).T.max(axis=1)

    return prediction, proba_


def phenotyping(data_df, n_components=3):
    df = data_df.copy()

    # Initialize CD45 and HNFalpha classifications with scores
    CD45_class, CD45_score = gmm_marker(
        data=df, marker="CD45", n_components=n_components
    )
    df["CD45_class"] = CD45_class
    df["CD45_score"] = CD45_score

    HNFalpha_class, HNFalpha_score = gmm_marker(
        data=df, marker="HNFalpha", n_components=n_components
    )
    df["HNFalpha_class"] = HNFalpha_class
    df["HNFalpha_score"] = HNFalpha_score

    immune_cell_lst = ["CD3", "CD15", "CD19", "CD56", "CD57", "CD16"]
    for marker in immune_cell_lst:
        marker_str_class = f"{marker}_class"
        marker_str_score = f"{marker}_score"
        df[marker_str_class] = -1
        df[marker_str_score] = 0.0

        valid_rows = (df["CD45_class"] == 1) & (df["HNFalpha_class"] == 0)
        if valid_rows.any():
            marker_class, marker_score = gmm_marker(
                data=df.loc[valid_rows], marker=marker, n_components=n_components
            )
            df.loc[valid_rows, marker_str_class] = marker_class
            df.loc[valid_rows, marker_str_score] = marker_score

    # Markers for CD3+ cells
    cd3_pos_markers = ["TCRVa", "CD161", "CD8", "CD4", "FoxP3", "CD25"]
    for marker in cd3_pos_markers:
        marker_str_class = f"{marker}_class"
        marker_str_score = f"{marker}_score"
        df[marker_str_class] = -1
        df[marker_str_score] = 0.0

        valid_rows = df["CD3_class"] == 1
        if valid_rows.any():
            marker_class, marker_score = gmm_marker(
                data=df.loc[valid_rows], marker=marker, n_components=n_components
            )
            df.loc[valid_rows, marker_str_class] = marker_class
            df.loc[valid_rows, marker_str_score] = marker_score

    # Markers for CD3- cells
    cd3_neg_markers = [
        "CD15",
        "CD19",
        "CD56",
        "CD57",
        "CD16",
        "CD66b",
        "CD68",
        "CD163",
        "CD11c",
        "HLADR",
    ]
    for marker in cd3_neg_markers:
        marker_str_class = f"{marker}_class"
        marker_str_score = f"{marker}_score"
        df[marker_str_class] = -1
        df[marker_str_score] = 0.0

        valid_rows = df["CD3_class"] == 0
        if valid_rows.any():
            marker_class, marker_score = gmm_marker(
                data=df.loc[valid_rows], marker=marker, n_components=n_components
            )
            df.loc[valid_rows, marker_str_class] = marker_class
            df.loc[valid_rows, marker_str_score] = marker_score

    # Markers for parenchymal cells
    parenchymal_markers = ["pancytokeratin", "EPCAM", "CD34", "aSMA", "LYVE-1"]
    for marker in parenchymal_markers:
        marker_str_class = f"{marker}_class"
        marker_str_score = f"{marker}_score"
        df[marker_str_class] = -1
        df[marker_str_score] = 0.0

        valid_rows = (df["CD45_class"] == 0) & (df["HNFalpha_class"] == 0)
        if valid_rows.any():
            marker_class, marker_score = gmm_marker(
                data=df.loc[valid_rows], marker=marker, n_components=n_components
            )
            df.loc[valid_rows, marker_str_class] = marker_class
            df.loc[valid_rows, marker_str_score] = marker_score

    return df


def annotate_cell_types(df):
    """
    Annotate cell types based on gating results and store the annotations
    in corresponding columns for each cell type, using the product of scores
    instead of a binary indicator.

    Parameters:
    df (pd.DataFrame): DataFrame containing gating results with columns like
                       "CD45_class", "HNFalpha_class", "CD3_class", etc.

    Returns:
    pd.DataFrame: The original DataFrame augmented with 17 new columns,
                  each representing a specific cell type, storing the product
                  of scores if matched.
    """

    # List of cell types to annotate
    cell_types = [
        "Immune cell",
        "Parenchymal cell",
        "T cell",
        "MAIT cell",
        "CD8+ T cell",
        "CD4+ T cell",
        "Treg",
        "B cell",
        "NK cell",
        "Neutrophil Granulocyte",
        "Kupffer cell",
        "M2 Macrophage",
        "Dendritic cell",
        "Hepatocytes/Tumor",
        "Biliary epithelial cells",
        "Vascular endothelial cells",
        "LSEC",
    ]

    # Initialize columns for each cell type to 0.0
    for cell_type in cell_types:
        df[cell_type] = 0.0

    # Define each cell type with the product of scores
    df.loc[(df["CD45_class"] == 1) & (df["HNFalpha_class"] == 0), "Immune cell"] = (
        df["CD45_score"] * df["HNFalpha_score"]
    )

    df.loc[(df["CD45_class"] == 0), "Parenchymal cell"] = df["CD45_score"]

    df.loc[
        (df["CD45_class"] == 1) & (df["HNFalpha_class"] == 0) & (df["CD3_class"] == 1),
        "T cell",
    ] = df["CD45_score"] * df["HNFalpha_score"] * df["CD3_score"]

    df.loc[
        (df["CD45_class"] == 1)
        & (df["HNFalpha_class"] == 0)
        & (df["CD3_class"] == 1)
        & (df["TCRVa_class"] == 1)
        & (df["CD161_class"] == 1),
        "MAIT cell",
    ] = (
        df["CD45_score"]
        * df["HNFalpha_score"]
        * df["CD3_score"]
        * df["TCRVa_score"]
        * df["CD161_score"]
    )

    df.loc[
        (df["CD45_class"] == 1)
        & (df["HNFalpha_class"] == 0)
        & (df["CD3_class"] == 1)
        & (df["CD8_class"] == 1)
        & (df["CD4_class"] == 0)
        & (df["TCRVa_class"] == 0),
        "CD8+ T cell",
    ] = df["CD45_score"] * df["HNFalpha_score"] * df["CD3_score"] * df["CD8_score"]

    df.loc[
        (df["CD45_class"] == 1)
        & (df["HNFalpha_class"] == 0)
        & (df["CD3_class"] == 1)
        & (df["CD8_class"] == 0)
        & (df["CD4_class"] == 1)
        & (df["TCRVa_class"] == 0)
        & (df["FoxP3_class"] == 0),
        "CD4+ T cell",
    ] = df["CD45_score"] * df["HNFalpha_score"] * df["CD3_score"] * df["CD4_score"]

    df.loc[
        (df["CD45_class"] == 1)
        & (df["HNFalpha_class"] == 0)
        & (df["CD3_class"] == 1)
        & (df["CD8_class"] == 0)
        & (df["CD4_class"] == 1)
        & (df["TCRVa_class"] == 0)
        & (df["FoxP3_class"] == 1)
        & (df["CD25_class"] == 1),
        "Treg",
    ] = (
        df["CD45_score"]
        * df["HNFalpha_score"]
        * df["CD3_score"]
        * df["CD4_score"]
        * df["FoxP3_score"]
        * df["CD25_score"]
    )

    df.loc[
        (df["CD45_class"] == 1)
        & (df["HNFalpha_class"] == 0)
        & (df["CD3_class"] == 0)
        & (df["CD15_class"] == 0)
        & (df["CD19_class"] == 1),
        "B cell",
    ] = (
        df["CD45_score"]
        * df["HNFalpha_score"]
        * (df["CD3_score"])
        * (df["CD15_score"])
        * df["CD19_score"]
    )

    df.loc[
        (df["CD45_class"] == 1)
        & (df["HNFalpha_class"] == 0)
        & (df["CD3_class"] == 0)
        & (df["CD56_class"] == 1),
        "NK cell",
    ] = df["CD45_score"] * df["HNFalpha_score"] * (df["CD3_score"]) * df["CD56_score"]

    df.loc[
        (df["CD45_class"] == 1)
        & (df["HNFalpha_class"] == 0)
        & (df["CD3_class"] == 0)
        & (df["CD15_class"] == 1)
        & (df["CD66b_class"] == 1),
        "Neutrophil Granulocyte",
    ] = (
        df["CD45_score"]
        * df["HNFalpha_score"]
        * (df["CD3_score"])
        * df["CD15_score"]
        * df["CD66b_score"]
    )

    df.loc[
        (df["CD45_class"] == 1)
        & (df["HNFalpha_class"] == 0)
        & (df["CD3_class"] == 0)
        & (df["CD68_class"] == 1)
        & (df["CD163_class"] == 0),
        "Kupffer cell",
    ] = (
        df["CD45_score"]
        * df["HNFalpha_score"]
        * (df["CD3_score"])
        * df["CD68_score"]
        * (df["CD163_score"])
    )

    df.loc[
        (df["CD45_class"] == 1)
        & (df["HNFalpha_class"] == 0)
        & (df["CD3_class"] == 0)
        & (df["CD163_class"] == 1),
        "M2 Macrophage",
    ] = df["CD45_score"] * df["HNFalpha_score"] * (df["CD3_score"]) * df["CD163_score"]

    df.loc[
        (df["CD45_class"] == 1)
        & (df["HNFalpha_class"] == 0)
        & (df["CD3_class"] == 0)
        & (df["CD11c_class"] == 1)
        & (df["HLADR_class"] == 1),
        "Dendritic cell",
    ] = (
        df["CD45_score"]
        * df["HNFalpha_score"]
        * (df["CD3_score"])
        * df["CD11c_score"]
        * df["HLADR_score"]
    )

    df.loc[
        (df["CD45_class"] == 0) & (df["HNFalpha_class"] == 1), "Hepatocytes/Tumor"
    ] = (df["CD45_score"]) * df["HNFalpha_score"]

    df.loc[
        (df["CD45_class"] == 0)
        & (df["HNFalpha_class"] == 0)
        & (df["pancytokeratin_class"] == 1)
        & (df["EPCAM_class"] == 1),
        "Biliary epithelial cells",
    ] = (
        (df["CD45_score"])
        * (df["HNFalpha_score"])
        * df["pancytokeratin_score"]
        * df["EPCAM_score"]
    )

    df.loc[
        (df["CD45_class"] == 0)
        & (df["HNFalpha_class"] == 0)
        & (df["CD34_class"] == 1)
        & (df["aSMA_class"] == 1)
        & (df["LYVE-1_class"] == 1),
        "Vascular endothelial cells",
    ] = (
        (df["CD45_score"])
        * (df["HNFalpha_score"])
        * df["CD34_score"]
        * df["aSMA_score"]
        * df["LYVE-1_score"]
    )

    df.loc[
        (df["CD45_class"] == 0)
        & (df["HNFalpha_class"] == 0)
        & (df["LYVE-1_class"] == 0),
        "LSEC",
    ] = (df["CD45_score"]) * (df["HNFalpha_score"]) * (df["LYVE-1_score"])

    # Optional: Add an "Unknown" column for cells that do not match any cell type
    df["Unknown"] = 0.0
    known_types = cell_types  # List of all known cell type columns
    condition = (df[known_types] == 0).all(axis=1)
    df.loc[condition, "Unknown"] = (
        1.0  # Or set to a probability-like value if preferred
    )

    return df


def select_phenotype(cell_counts_df):
    """
    Selects the most relevant phenotype for each row in a DataFrame of cell type counts
    based on a predefined priority ranking.

    Parameters:
        cell_counts_df (pd.DataFrame): A pandas DataFrame where columns are cell types
                                       and rows are samples. The values are cell type counts.

    Returns:
        pd.Series: A pandas Series where the index matches the input DataFrame's index,
                   and the values are the selected phenotypes for each sample.
    """
    # Define the priority level for each cell type
    cell_type_priority = {
        "Immune cell": 1,
        "Parenchymal cell": 1,
        "Unknown": 1,
        "Hepatocytes/Tumor": 2,
        "Biliary epithelial cells": 2,
        "Vascular endothelial cells": 2,
        "LSEC": 2,
        "T cell": 2,
        "Treg": 3,
        "MAIT cell": 3,
        "CD8+ T cell": 3,
        "CD4+ T cell": 3,
        "B cell": 2,
        "NK cell": 2,
        "Neutrophil Granulocyte": 2,
        "Kupffer cell": 2,
        "M2 Macrophage": 2,
        "Dendritic cell": 2,
    }

    # Convert priority levels to a NumPy array aligned with the DataFrame columns
    priorities = np.array(
        [cell_type_priority.get(col, 0) for col in cell_counts_df.columns]
    )

    # Mask cell types with zero counts
    non_zero_mask = cell_counts_df.values != 0

    # Apply priority levels only to non-zero counts
    priority_matrix = np.where(non_zero_mask, priorities, -np.inf)

    # Find the maximum priority for each row
    max_priority = np.max(priority_matrix, axis=1)

    # Create a mask for columns that match the maximum priority
    highest_priority_mask = priority_matrix == max_priority[:, np.newaxis]

    # For each row, select the cell type with the highest count among those with the highest priority
    # If multiple have the same count, the first one will be chosen
    counts_with_priority = np.where(
        highest_priority_mask, cell_counts_df.values, -np.inf
    )
    best_match_indices = np.argmax(counts_with_priority, axis=1)

    # Map indices back to cell type names
    selected_phenotypes = pd.Series(
        cell_counts_df.columns[best_match_indices], index=cell_counts_df.index
    )

    return selected_phenotypes
