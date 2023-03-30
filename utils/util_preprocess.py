import json
import os
import re
import typing

import numpy as np
import pandas as pd


class DataFramePreprocess:
    def __init__(self, df: pd.DataFrame, config_name: str):
        assert config_name is not None
        # dataframe
        self.df = df

        config_path = os.path.join("config", config_name + ".json")
        with open(config_path, mode="r") as f:
            content = json.load(f)

        assert isinstance(content, dict)
        # TODO handle exceptions
        for k, v in content.items():
            setattr(self, k, v)

    def filter_unwanted_columns(self):
        print("Removing unwanted columns and duplicates...")
        # log the number of rows before removing unwanted columns
        print(
            f"Shape before removing unwanted columns and duplicates: {self.df.shape}"
        )
        self.df = self.df[self.keep_cols].drop_duplicates().reset_index(drop=True)
        print(
            f"Shape after removing unwanted columns and duplicates: {self.df.shape}\n"
        )

    def filter_unwanted_rows(self, filter_condition: typing.Dict):
        print("Removing unwanted rows and duplicates...")
        # log the number of rows before removing unwanted rows
        print(
            f"Shape before removing unwanted rows and duplicates: {self.df.shape}"
        )

        idx = pd.Series([True] * self.df.shape[0])
        for col, pat in filter_condition.items():
            idx = idx & pat(self.df[col])

        self.df = self.df[idx].drop_duplicates().reset_index(drop=True)

        print(
            f"Shape after removing unwanted rows and duplicates: {self.df.shape}\n"
        )
        return self.df

    def process_misc_price(self, series: pd.Series) -> pd.Series:
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
                series, idx = self.extract_replace(complete_regex(c), series)
                extracted_idx = extracted_idx | idx

        # set to nan the rows that are not extracted
        series.loc[~extracted_idx] = np.nan
        # convert to float
        series = series.astype(float)

        print(f"Number of rows without value: {series.isna().sum()}")
        return series

    def process_float_feature(self, series: pd.Series, reg: str) -> pd.Series:
        # extract the pattern
        series, idx = self.extract_replace(reg, series)
        # set to nan the rows that are not extracted
        series.loc[~idx] = np.nan
        # replace the comma with dot
        series = series.str.replace(",", ".", regex=False)
        # convert to float
        series = series.astype(float)

        print(f"Number of rows without a value: {series.isna().sum()}")
        return series

    def process_binary_feature(
        self, series_raw: pd.Series, pat: str, repl: typing.List[str]
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
        self, series: pd.Series, pat: str, repl_cols: typing.Dict
    ) -> pd.DataFrame:
        # extract the pattern
        extract_col = series.str.extract(pat, flags=re.IGNORECASE, expand=True)
        # rename the columns
        extract_col = extract_col.rename(columns=repl_cols)
        # convert the height and width to float
        extract_col = extract_col.astype(float)

        print(f"Number of rows without a value: {series.isna().sum()}")
        return extract_col

    def process_multi_feature(
        self, series: pd.Series, pats_repls: typing.Dict
    ) -> pd.Series:
        extracted_idx = pd.Series(False, index=series.index)
        for pat, repl in pats_repls.items():
            series, idx = self.extract_replace(pat, series)
            series.loc[idx] = repl
            extracted_idx = extracted_idx | idx

        # set to nan the rows that are not extracted
        series.loc[~extracted_idx] = np.nan

        print(f"Number of rows without value: {series.isna().sum()}")
        return series

    def process_camera_features(self, df: pd.DataFrame, col: str):
        assert col in self.camera_features.keys()

        cols = self.camera_features[col]["cols"]
        pat = self.camera_features[col]["pat"]
        mode = col.split("_")[0]
        camera_res, num_camera = f"{mode}_camera_resolution", f"num_{mode}_camera"

        df_new = pd.DataFrame(
            np.nan, index=df.index, columns=[camera_res, num_camera], dtype=float
        )

        extracted_idx = pd.Series(False, index=df.index)
        for (i, col) in enumerate(cols):
            series_mp, idx = self.extract_replace(pat, df[col])
            df_new.loc[idx, camera_res] = series_mp.loc[idx]
            extracted_idx = extracted_idx | idx

            if i == len(cols) - 1 and mode == "main":
                df_new.loc[idx, num_camera] = 2
            else:
                df_new.loc[idx, num_camera] = i + 1

        # set to nan the rows that are not extracted
        df_new.loc[~extracted_idx, camera_res] = np.nan

        print(f"Number of rows without value: {df_new.isna().sum()}")
        return df_new

    def extract_replace(self, pattern: str, series_raw: pd.Series):
        series = series_raw.copy()
        # extract the pattern
        extract_col = series.str.extract(pattern, expand=False, flags=re.IGNORECASE)
        extract_idx = extract_col.notna()
        # replace the extracted pattern with the extracted value
        series.loc[extract_idx] = extract_col

        print(f"Number of rows extracted: {extract_idx.sum()}")
        return series, extract_idx


class GSMArenaPreprocess(DataFramePreprocess):
    def __init__(self, df: pd.DataFrame, config_name: str = "gsmarena"):
        super().__init__(df, config_name)

    def preprocess(self):
        self.filter_unwanted_columns()
        self.preprocess_feature("misc_price")
        self.preprocess_feature("launch_announced")
        self.preprocess_feature("display_size")
        self.preprocess_feature("display_type")
        self.preprocess_feature("display_resolution")
        self.preprocess_feature("memory_card_slot")
        self.preprocess_feature("sound_loudspeaker")
        self.preprocess_feature("sound_3.5mm_jack")
        self.preprocess_feature("comms_wlan")
        self.preprocess_feature("comms_nfc")
        self.preprocess_feature("memory_internal")
        self.preprocess_feature("network_technology")
        self.preprocess_feature("battery")
        self.preprocess_feature("battery_charging")
        self.preprocess_feature("body")
        self.preprocess_feature("main_camera_cols")
        self.preprocess_feature("selfie_camera_cols")
        return self.df    


    def preprocess_feature(self, col_name: str) -> pd.DataFrame:
        print(f"Preprocessing {col_name}...")
        print(
            f"Number of rows before extracting valid {col_name}: {self.df.shape[0]}"
        )

        if col_name in self.camera_features.keys():
            feat_cols = self.camera_features[col_name]["cols"]
        else:
            feat_cols = [col_name]

        # fill the nan values of only the columns that are in the fill_cols dict
        if col_name in self.fill_cols.keys():
            self.df.loc[:, feat_cols] = self.df.loc[:, feat_cols].fillna(
                self.fill_cols[col_name]
            )

        self.df = self.df.dropna(subset=feat_cols, how="all").reset_index(drop=True)
        print(
            f"Number of rows after removing null values in {col_name}: {self.df.shape[0]}"
        )

        # preprocess the column based on the column name
        if col_name == "misc_price":
            self.df.loc[:, col_name] = self.process_misc_price(self.df.loc[:, col_name])
        elif col_name in self.float_features.keys():
            self.df.loc[:, col_name] = self.process_float_feature(
                self.df.loc[:, col_name], self.float_features[col_name]
            )
        elif col_name in self.binary_features.keys():
            self.df.loc[:, col_name] = self.process_binary_feature(
                self.df.loc[:, col_name],
                self.binary_features[col_name]["pat"],
                self.binary_features[col_name]["repl"],
            )
            # rename the column
            self.df = self.df.rename(
                columns={col_name: self.binary_features[col_name]["new_col"]}
            )
            col_name = self.binary_features[col_name]["new_col"]
            feat_cols = [col_name]
        elif col_name in self.multi_col_features.keys():
            df_feats = self.process_multi_col_features(
                self.df.loc[:, col_name],
                self.multi_col_features[col_name]["pat"],
                self.multi_col_features[col_name]["new_cols"],
            )

            # concat df_feats with df
            self.df = pd.concat([self.df, df_feats], axis=1)
            self.df = self.df.drop(columns=feat_cols)

            # update the col_name and feat_cols to the new columns
            col_name = df_feats.columns.to_list()
            feat_cols = col_name
        elif col_name in self.multi_features.keys():
            self.df.loc[:, col_name] = self.process_multi_feature(
                self.df.loc[:, col_name], self.multi_features[col_name]
            )
        elif col_name in ["main_camera_cols", "selfie_camera_cols"]:
            df_feat = self.process_camera_features(self.df.loc[:, feat_cols], col_name)

            # update the df with the new columns and drop the old columns
            self.df = pd.concat([self.df, df_feat], axis=1)
            self.df = self.df.drop(columns=feat_cols)

            # update the col_name and feat_cols to the new columns
            col_name = df_feat.columns.to_list()
            feat_cols = col_name
        else:
            print(f"Column {col_name} not processed!")
            raise NotImplementedError

        # remove the rows that are nan after preprocessing
        self.df = self.df.dropna(subset=feat_cols, how="all").reset_index(drop=True)
        print(
            f"Number of rows after extracting valid {col_name}: {self.df.shape[0]}\n"
        )
