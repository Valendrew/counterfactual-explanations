import re
import typing

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.info, format="%(message)s")

keep_cols = [
    "misc_price",
    "launch_announced",
    "display_size",
    "display_type",
    "display_resolution",
    "memory_card_slot",
    "sound_loudspeaker",
    "sound_3.5mm_jack",
    "comms_wlan",
    "comms_nfc",
    "memory_internal",
    "network_technology",
    "battery",
    "battery_charging",
    "body",
    "main_camera_single",
    "main_camera_dual",
    "main_camera_quad",
    "main_camera_triple",
    "main_camera_five",
    "main_camera_dual_or_triple",
    "selfie_camera_single",
    "selfie_camera_dual",
    "selfie_camera_triple",
]

# TODO group in classes
# Float features
float_features = {
    "launch_announced": r"([\d]{4})",
    "display_size": r"([\d]{1,2}\.[\d]{1,2}) inches",
    "battery": r"([\d]{1,2},?[\d]{3})(?:/?[\d]+)?\s?mah",
}

# Binary features
binary_features = {
    "display_type": {"repl": [1, 0], "new_col": "has_oled_display", "pat": r"oled"},
    "memory_card_slot": {
        "repl": [0, 1],
        "new_col": "has_memory_card_slot",
        "pat": r"\bno\b",
    },
    "sound_loudspeaker": {
        "repl": [1, 0],
        "new_col": "has_stereo_speakers",
        "pat": r"stereo|dual|multiple|quad",
    },
    "sound_3.5mm_jack": {
        "repl": [1, 0],
        "new_col": "has_3.5mm_jack",
        "pat": r"\byes\b",
    },
    "comms_wlan": {"repl": [0, 1], "new_col": "has_wlan_5ghz", "pat": r"[^/]b/g/n[^/]"},
    "comms_nfc": {"repl": [0, 1], "new_col": "has_nfc", "pat": r"\bno\b"},
    "battery_charging": {
        "repl": [1, 0],
        "new_col": "has_wireless_charging",
        "pat": r"wireless",
    },
    "body": {
        "repl": [1, 0],
        "new_col": "is_waterproof",
        "pat": r"splash|water|ip[6-9]",
    },
}

# Multiple features
multi_features = {
    "network_technology": {"(5G)": "5G", "(LTE)": "4G", "(UMTS)": "3G", "(HSPA)": "3G"}
}


# Multi columns features
multi_col_features = {
    "display_resolution": {
        "pat": r"(?P<w>[\d]{3,4}) x (?P<h>[\d]{3,4})",  # TODO add ratio
        "new_cols": {"w": "display_width", "h": "display_height"},
    },
    "memory_internal": {
        "pat": r"^(?P<rom>[\d]{1,3}(?:[\.,][\d]{2})?).*?GB\D*?(?P<ram>[\d]{1,2}(?:[\.,][\d]{2})?).*?GB\s*RAM",
        "new_cols": {"rom": "memory_rom_gb", "ram": "memory_ram_gb"},
    },
}

# Camera columns
camera_features = {
    "main_camera_cols": {
        "pat": r"^([\d]{1,3}(?:[\.][\d]{1,2})?)(?:/?[\d]+)?\s?mp",
        "cols": [
            "main_camera_single",
            "main_camera_dual",
            "main_camera_triple",
            "main_camera_quad",
            "main_camera_five",
            "main_camera_dual_or_triple",
        ],
    },
    "selfie_camera_cols": {
        "pat": r"^([\d]{1,3}(?:[\.][\d]{1,2})?)(?:/?[\d]+)?\s?mp",
        "cols": [
            "selfie_camera_single",
            "selfie_camera_dual",
            "selfie_camera_triple",
        ],
    },
}

# Filled columns
fill_cols = {"comms_nfc": "no", "battery_charging": "no", "body": "no"}


def filter_unwanted_columns(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Removing unwanted columns and duplicates...")
    # log the number of rows before removing unwanted columns
    logger.info(f"Shape before removing unwanted columns and duplicates: {df.shape}")
    df = df[keep_cols].drop_duplicates().reset_index(drop=True)
    logger.info(f"Shape after removing unwanted columns and duplicates: {df.shape}\n")
    return df


def filter_unwanted_rows(
    df: pd.DataFrame, filter_condition: typing.Dict
) -> pd.DataFrame:
    logger.info("Removing unwanted rows and duplicates...")
    # log the number of rows before removing unwanted rows
    logger.info(f"Shape before removing unwanted rows and duplicates: {df.shape}")

    idx = pd.Series([True] * df.shape[0])
    for col, pat in filter_condition.items():
        idx = idx & pat(df[col])

    df = df[idx].drop_duplicates().reset_index(drop=True)

    logger.info(f"Shape after removing unwanted rows and duplicates: {df.shape}\n")
    return df


def preprocess_feature(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    logger.debug(f"Preprocessing {col_name}...")
    logger.info(f"Number of rows before extracting valid {col_name}: {df.shape[0]}")

    if col_name in camera_features.keys():
        feat_cols = camera_features[col_name]["cols"]
    else:
        feat_cols = [col_name]

    # fill the nan values of only the columns that are in the fill_cols dict
    if col_name in fill_cols.keys():
        df.loc[:, feat_cols] = df.loc[:, feat_cols].fillna(fill_cols[col_name])

    df = df.dropna(subset=feat_cols, how="all").reset_index(drop=True)
    logger.debug(
        f"Number of rows after removing null values in {col_name}: {df.shape[0]}"
    )

    # preprocess the column based on the column name
    if col_name == "misc_price":
        df.loc[:, col_name] = process_misc_price(df.loc[:, col_name])
    elif col_name in float_features.keys():
        df.loc[:, col_name] = process_float_feature(
            df.loc[:, col_name], float_features[col_name]
        )
    elif col_name in binary_features.keys():
        df.loc[:, col_name] = process_binary_feature(
            df.loc[:, col_name],
            binary_features[col_name]["pat"],
            binary_features[col_name]["repl"],
        )
        # rename the column
        df = df.rename(columns={col_name: binary_features[col_name]["new_col"]})
        col_name = binary_features[col_name]["new_col"]
        feat_cols = [col_name]
    elif col_name in multi_col_features.keys():
        df_feats = process_multi_col_features(
            df.loc[:, col_name],
            multi_col_features[col_name]["pat"],
            multi_col_features[col_name]["new_cols"],
        )

        # concat df_feats with df
        df = pd.concat([df, df_feats], axis=1)
        df = df.drop(columns=feat_cols)

        # update the col_name and feat_cols to the new columns
        col_name = df_feats.columns.to_list()
        feat_cols = col_name
    elif col_name in multi_features.keys():
        df.loc[:, col_name] = process_multi_feature(
            df.loc[:, col_name], multi_features[col_name]
        )
    elif col_name in ["main_camera_cols", "selfie_camera_cols"]:
        df_feat = process_camera_features(df.loc[:, feat_cols], col_name)

        # update the df with the new columns and drop the old columns
        df = pd.concat([df, df_feat], axis=1)
        df = df.drop(columns=feat_cols)

        # update the col_name and feat_cols to the new columns
        col_name = df_feat.columns.to_list()
        feat_cols = col_name
    else:
        logger.warning(f"Column {col_name} not processed!")
        raise NotImplementedError

    # remove the rows that are nan after preprocessing
    df = df.dropna(subset=feat_cols, how="all").reset_index(drop=True)
    logger.info(f"Number of rows after extracting valid {col_name}: {df.shape[0]}\n")

    return df


def process_misc_price(series: pd.Series) -> pd.Series:
    # replace thin space with space
    to_replace = {
        "<e2><80><89>": " ",
        "<e2><82><ac>": "EUR",
        "€": "EUR",
        "\$": "USD",
        "<c2><a3>": "GBP",
        "(,)(?=[\d]{3})": "",
    }

    # replace UTF-8 hexadecimals into specific characters
    # TODO replace without explictly mentioning the characters
    for k, v in to_replace.items():
        series = series.str.replace(k, v, regex=True, flags=re.IGNORECASE)

    # numeric_regex for extracting prices
    numeric_regex = r"[\d]{2,5}(?:\.[\d]{1,2})?"
    # list of currencies supported
    currencies = ["EUR", "USD", "GBP"]
    # typesof regex with the currency at the start or at the end
    complete_regex_start = lambda c: f"{c}\s?({numeric_regex})"
    complete_regex_end = lambda c: f"About\s?({numeric_regex})\s?{c}"

    # extract the numeric value and currency
    # create series of false values
    extracted_idx = pd.Series(False, index=series.index)
    for c in currencies:
        for complete_regex in [complete_regex_start, complete_regex_end]:
            series, idx = extract_replace(complete_regex(c), series)
            extracted_idx = extracted_idx | idx

    # set to nan the rows that are not extracted
    series.loc[~extracted_idx] = np.nan
    # convert to float
    series = series.astype(float)

    logger.debug(f"Number of rows without value: {series.isna().sum()}")
    return series


def process_float_feature(series: pd.Series, reg: str) -> pd.Series:
    # extract the pattern
    series, idx = extract_replace(reg, series)
    # set to nan the rows that are not extracted
    series.loc[~idx] = np.nan
    # replace the comma with dot
    series = series.str.replace(",", ".", regex=False)
    # convert to float
    series = series.astype(float)

    logger.debug(f"Number of rows without a value: {series.isna().sum()}")
    return series


def process_binary_feature(
    series_raw: pd.Series, pat: str, repl: typing.List[str]
) -> pd.Series:
    """Process the binary feature by replacing the pattern with the first element of repl
    and the rest with the second element of repl.

    Args:
        series_raw (pd.Series): the series to process
        pat (str): the pattern to replace
        repl (typing.List[str]): the replacement values

    Returns:
        pd.Series: the processed series
    """
    series = series_raw.copy()
    idx = series.str.contains(pat, flags=re.IGNORECASE, regex=True)
    series.loc[idx] = repl[0]
    series.loc[~idx] = repl[1]
    return series


def process_multi_col_features(
    series: pd.Series, pat: str, repl_cols: typing.Dict
) -> pd.DataFrame:
    # extract the pattern
    extract_col = series.str.extract(pat, flags=re.IGNORECASE, expand=True)
    # rename the columns
    extract_col = extract_col.rename(columns=repl_cols)
    # convert the height and width to float
    extract_col = extract_col.astype(float)

    logger.debug(f"Number of rows without a value: {series.isna().sum()}")
    return extract_col


def process_multi_feature(series: pd.Series, pats_repls: typing.Dict) -> pd.Series:
    extracted_idx = pd.Series(False, index=series.index)
    for pat, repl in pats_repls.items():
        series, idx = extract_replace(pat, series)
        series.loc[idx] = repl
        extracted_idx = extracted_idx | idx

    # set to nan the rows that are not extracted
    series.loc[~extracted_idx] = np.nan

    logger.debug(f"Number of rows without value: {series.isna().sum()}")
    return series


def process_camera_features(df: pd.DataFrame, col: str):
    assert col in camera_features.keys()

    cols = camera_features[col]["cols"]
    pat = camera_features[col]["pat"]
    mode = col.split("_")[0]
    camera_res, num_camera = f"{mode}_camera_resolution", f"num_{mode}_camera"

    df_new = pd.DataFrame(
        np.nan, index=df.index, columns=[camera_res, num_camera], dtype=float
    )

    extracted_idx = pd.Series(False, index=df.index)
    for (i, col) in enumerate(cols):
        series_mp, idx = extract_replace(pat, df[col])
        df_new.loc[idx, camera_res] = series_mp.loc[idx]
        extracted_idx = extracted_idx | idx

        if i == len(cols) - 1 and mode == "main":
            df_new.loc[idx, num_camera] = 2
        else:
            df_new.loc[idx, num_camera] = i + 1

    # set to nan the rows that are not extracted
    df_new.loc[~extracted_idx, camera_res] = np.nan

    logger.debug(f"Number of rows without value: {df_new.isna().sum()}")
    return df_new


def extract_replace(pattern: str, series_raw: pd.Series):
    series = series_raw.copy()
    # extract the pattern
    extract_col = series.str.extract(pattern, expand=False, flags=re.IGNORECASE)
    extract_idx = extract_col.notna()
    # replace the extracted pattern with the extracted value
    series.loc[extract_idx] = extract_col

    logger.debug(f"Extracted {extract_idx} rows")
    return series, extract_idx
