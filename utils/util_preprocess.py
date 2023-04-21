import json
import math
import os
import re
import typing

import numpy as np
import pandas as pd


class DataTransformer:
    @staticmethod
    def filter_unwanted_columns(
        df: pd.DataFrame, keep_cols: typing.List[str], dup_cols=None, verbose=False
    ) -> pd.DataFrame:
        """Filter the unwanted columns and duplicates.

        Args:
            df (pd.DataFrame): dataframe to filter
            keep_cols (list): list of columns to keep

        Returns:
            pd.DataFrame: filtered dataframe
        """
        vvprint = print if verbose else lambda *a, **k: None
        vvprint("Removing unwanted columns and duplicates...")
        vvprint(f"Shape before removing unwanted columns and duplicates: {df.shape}")

        if dup_cols is None:
            dup_cols = keep_cols

        df = df[keep_cols].drop_duplicates(subset=dup_cols, ignore_index=True)

        vvprint(f"Shape after removing unwanted columns and duplicates: {df.shape}\n")
        return df

    @staticmethod
    def filter_unwanted_rows(
        df: pd.DataFrame, filter_condition: typing.Dict, dup_cols=None, verbose=False
    ) -> pd.DataFrame:
        """Filter the unwanted rows and duplicates.

        Args:
            df (pd.DataFrame): dataframe to filter
            filter_condition (typing.Dict): dictionary of filter conditions
            verbose (bool, optional): verbose mode. Defaults to False.

        Returns:
            pd.DataFrame: filtered dataframe
        """
        vvprint = print if verbose else lambda *a, **k: None

        vvprint("Removing unwanted rows and duplicates...")
        vvprint(f"Shape before removing unwanted rows and duplicates: {df.shape}")

        if dup_cols is None:
            dup_cols = df.columns

        idx = pd.Series([True] * df.shape[0])
        # iterate over the filter conditions and apply the function to the column
        for col, pat in filter_condition:
            idx = idx & pat(df[col])

        # drop the unwanted rows and duplicates
        df = df[idx].drop_duplicates(dup_cols, ignore_index=True)

        vvprint(f"Shape after removing unwanted rows and duplicates: {df.shape}\n")
        return df

    @classmethod
    def process_misc_price(
        cls, series: pd.Series, conversion_rates: typing.Dict, verbose=False
    ) -> pd.Series:
        """Process the miscellaneous price feature. It extracts the price from the string and converts it to euro.

        Args:
            series (pd.Series): series to process
            conversion_rates (typing.Dict): dictionary of conversion rates to convert the price to euro

        Returns:
            pd.Series: processed series
        """
        vvprint = print if verbose else lambda *a, **k: None

        """ Replace the comma of thousand with dot, currency symbol with the currency name and
        the thin space codes with a space """
        chars_replacements = {
            "<e2><80><89>": " ",
            "<e2><82><ac>": "EUR",
            "<e2><82><B9>": "INR",
            "<c2><a3>": "GBP",
            "€": "EUR",
            "\$": "USD",
            "£": "GBP",
            "" "\u2009": " ",
            "(,)(?=[\d]{3})": "",
        }

        for k, v in chars_replacements.items():
            series = series.str.replace(k, v, regex=True, flags=re.IGNORECASE)

        # numeric_regex to extract prices
        numeric_regex = r"[\d]{2,6}(?:\.[\d]{1,2})?"
        # list of currencies supported
        currencies = ["EUR", "USD", "INR", "GBP"]
        # type of regex with the currency at the start or at the end
        complete_regex_start = lambda c: f"{c}\s?({numeric_regex})"
        complete_regex_end = lambda c: f"About\s?({numeric_regex})\s?{c}"

        """ Create series of false values to keep track of the extracted rows
        in order to set to nan the rows that are not extracted"""
        extracted_idx = pd.Series(False, index=series.index)
        # iterate over the currencies and the regex types
        for c in currencies:
            for complete_regex in [complete_regex_start, complete_regex_end]:
                # extract the price and replace the original value in the series
                series_raw, idx = cls.extract_replace(
                    complete_regex(c), series, verbose
                )
                # adjust the price to the euro if the currency is different from euro
                if c != "EUR":
                    series_raw.loc[idx] = (
                        series_raw.loc[idx]
                        .apply(lambda x: float(x) / conversion_rates[c])
                        .astype(str)
                    )

                series = series_raw
                # update the extracted_idx
                extracted_idx = extracted_idx | idx

        # set to nan the rows that are not extracted
        series.loc[~extracted_idx] = np.nan
        # convert to float
        series = series.astype(float)

        vvprint(f"Number of rows without value: {series.isna().sum()}")
        return series

    @classmethod
    def process_float_feature(
        cls, series: pd.Series, reg: str, verbose=False
    ) -> pd.Series:
        """Process the float feature by extracting the pattern and converting it to float.

        Args:
            series (pd.Series): series to process
            reg (str): regex to extract the pattern

        Returns:
            pd.Series: processed series
        """
        vvprint = print if verbose else lambda *a, **k: None
        # extract the pattern
        series, idx = cls.extract_replace(reg, series, verbose)
        # set to nan the rows that are not extracted
        series.loc[~idx] = np.nan
        # replace the comma with dot
        series = series.str.replace(",", ".", regex=False)
        # convert to float
        series = series.astype(float)

        vvprint(f"Number of rows without a value: {series.isna().sum()}")
        return series

    @classmethod
    def process_binary_feature(
        cls, series_raw: pd.Series, pat: str, repl: typing.List[str], verbose=False
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
        # extract the pattern
        idx = series.str.contains(pat, flags=re.IGNORECASE, regex=True)

        # create an array of zeros and replace the values with the replacement values
        idx = idx.to_numpy(dtype=bool)
        repl_array = np.zeros(series.size, dtype=int)
        repl_array[idx] = repl[0]
        repl_array[~idx] = repl[1]

        return pd.Series(repl_array, index=series.index)

    @classmethod
    def process_multi_col_features(
        cls, series: pd.Series, pat: str, repl_cols: typing.Dict, verbose=False
    ) -> pd.DataFrame:
        """Process the multi column features by extracting the patterns in a dataframe
        and renaming the columns with the keys of repl_cols.

        Args:
            series (pd.Series): series to process
            pat (str): regex to extract the pattern
            repl_cols (typing.Dict): dictionary of columns to rename
            verbose (bool, optional): verbose mode. Defaults to False.

        Returns:
            pd.DataFrame: processed dataframe with the extracted columns
        """
        vvprint = print if verbose else lambda *a, **k: None
        # extract the pattern
        extract_col = series.str.extract(pat, flags=re.IGNORECASE, expand=True)
        # rename the columns
        extract_col = extract_col.rename(columns=repl_cols)
        # convert the height and width to float
        extract_col = extract_col.astype(float)

        vvprint(
            f"Number of rows without a value: {extract_col.isna().any(axis=1).sum()}"
        )
        return extract_col

    @classmethod
    def process_multi_feature(
        cls, series: pd.Series, pats_repls: typing.Dict, verbose=False
    ) -> pd.Series:
        """Process the multi feature by extracting the patterns and replacing them with the
        values of the dictionary.

        Args:
            series (pd.Series): series to process
            pats_repls (typing.Dict): dictionary of patterns to replace
            verbose (bool, optional): verbose prints. Defaults to False.

        Returns:
            pd.Series: processed series
        """
        vvprint = print if verbose else lambda *a, **k: None

        extracted_idx = pd.Series(False, index=series.index)
        # iterate over the patterns and the replacements, extract the pattern and replace it
        for pat, repl in pats_repls.items():
            series, idx = cls.extract_replace(pat, series, verbose)
            series.loc[idx] = repl
            extracted_idx = extracted_idx | idx

        # set to nan the rows that are not extracted
        series.loc[~extracted_idx] = np.nan

        vvprint(f"Number of rows without value: {series.isna().sum()}")
        return series

    @classmethod
    def process_camera_features(
        cls, df: pd.DataFrame, col: str, camera_cols, pat, verbose=False
    ) -> pd.DataFrame:
        """Process the camera features by extracting the pattern and creating two new columns
        with the camera resolution and the number of cameras.

        Args:
            df (pd.DataFrame): dataframe to process
            col (str): general column name which defines the mode (main or selfie)
            camera_cols (_type_): camera columns to process
            pat (_type_): regex to extract the pattern
            verbose (bool, optional): verbose prints. Defaults to False.

        Returns:
            pd.DataFrame: dataframe with the new columns
        """
        vvprint = print if verbose else lambda *a, **k: None

        # retrieve whether the camera is main or selfie
        mode = col.split("_")[0]
        # new columns to create
        camera_res, num_camera = f"{mode}_camera_resolution", f"num_{mode}_camera"

        # new dataframe to create with the extracted columns
        df_new = pd.DataFrame(
            np.nan, index=df.index, columns=[camera_res, num_camera], dtype=float
        )

        extracted_idx = pd.Series(False, index=df.index)
        for (i, col) in enumerate(camera_cols):
            # replace the VGA with 0.3 MP
            df[col] = df[col].str.replace("VGA", "0.3 MP", regex=False, case=False)

            # extract the pattern of the camera resolution
            series, idx = cls.extract_replace(pat, df[col], verbose)
            df_new.loc[idx, camera_res] = series.loc[idx]
            extracted_idx = extracted_idx | idx

            # set the number of the camera
            if i == len(camera_cols) - 1 and mode == "main":
                df_new.loc[idx, num_camera] = 2
            else:
                df_new.loc[idx, num_camera] = i + 1

        # set to nan the rows that are not extracted
        df_new.loc[~extracted_idx, [camera_res, num_camera]] = np.nan

        vvprint(f"Number of rows without value: {df_new.isna().any(axis=1).sum()}")
        return df_new

    @staticmethod
    def extract_replace(
        pattern: str, series_raw: pd.Series, verbose=False
    ) -> pd.Series:
        """Extract the pattern from the series and replace it with the extracted value.

        Args:
            pattern (str): regex to extract the pattern
            series_raw (pd.Series): series to process
            verbose (_type_): verbose prints. Defaults to False.

        Returns:
            pd.Series: processed series
        """
        vvprint = print if verbose else lambda *a, **k: None

        series = series_raw.copy()
        # extract the pattern
        extract_col = series.str.extract(pattern, expand=False, flags=re.IGNORECASE)
        extract_idx = extract_col.notna()
        # replace the extracted pattern with the extracted value
        series.loc[extract_idx] = extract_col

        vvprint(f"Number of rows extracted: {extract_idx.sum()}")
        return series, extract_idx


class GSMArenaPreprocess:
    def __init__(self, config_name: str = "gsmarena"):
        """Constructor of the GSMArenaPreprocess class.

        Args:
            config_name (str, optional): the name of the config file. Defaults to "gsmarena".
        """
        # check if the config ends with .json
        if config_name.endswith(".json"):
            config_name = config_name[:-5]
        # load the config file only if it exists
        config_path = os.path.join("config", config_name + ".json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")
        with open(config_path, mode="r") as f:
            content = json.load(f)

        # check if the content is a dictionary
        assert isinstance(content, dict)
        # set the attributes of the class from the config file
        for k, v in content.items():
            setattr(self, k, v)

    def __call__(
        self, df: pd.DataFrame, filter_condition, verbose=False
    ) -> pd.DataFrame:
        """Preprocess the dataframe.

        Args:
            df (pd.DataFrame): the dataframe to preprocess

        Returns:
            pd.DataFrame: the preprocessed dataframe
        """
        return (
            df.pipe(
                DataTransformer.filter_unwanted_columns,
                self.unwanted_columns,
                verbose=verbose,
            )
            .pipe(self.preprocess_feature, "misc_price", verbose)
            .pipe(self.preprocess_feature, "oem_model", verbose)
            .pipe(self.preprocess_feature, "launch_announced", verbose)
            .pipe(self.preprocess_feature, "display_size", verbose)
            .pipe(self.preprocess_feature, "battery", verbose)
            .pipe(self.preprocess_feature, "display_type", verbose)
            .pipe(self.preprocess_feature, "memory_card_slot", verbose)
            .pipe(self.preprocess_feature, "sound_loudspeaker", verbose)
            .pipe(self.preprocess_feature, "sound_3.5mm_jack", verbose)
            .pipe(self.preprocess_feature, "comms_wlan", verbose)
            .pipe(self.preprocess_feature, "comms_nfc", verbose)
            .pipe(self.preprocess_feature, "battery_charging", verbose)
            .pipe(self.preprocess_feature, "body", verbose)
            .pipe(self.preprocess_feature, "display_resolution", verbose)
            .pipe(self.preprocess_feature, "memory_internal", verbose)
            .pipe(self.preprocess_feature, "network_technology", verbose)
            .pipe(self.preprocess_feature, "main_camera_cols", verbose)
            .pipe(self.preprocess_feature, "selfie_camera_cols", verbose)
            .pipe(
                DataTransformer.filter_unwanted_rows,
                filter_condition,
                verbose=verbose,
            )
        )

    def preprocess_feature(
        self, df: pd.DataFrame, col_name: str, verbose=False
    ) -> pd.DataFrame:
        """Preprocess a single feature. It is used to preprocess the features that are not
        extracted with the same method.

        Args:
            df (pd.DataFrame): the dataframe to preprocess
            col_name (str): the name of the column to preprocess

        Raises:
            NotImplementedError: if the column is not in the config file raise an error

        Returns:
            pd.DataFrame: the preprocessed dataframe
        """
        vvprint = print if verbose else lambda *a, **k: None
        vvprint(f"Preprocessing {col_name}...")
        vvprint(f"Number of rows before extracting valid {col_name}: {df.shape[0]}")

        """ Get the actual column names of the feature to preprocess
        The column names for the camera features aren't present in the dataframe so we need to
        get the actual column names from the config file."""
        if col_name in self.camera_features.keys():
            feat_cols = self.camera_features[col_name]["cols"]
        elif col_name in self.concat_features.keys():
            feat_cols = self.concat_features[col_name]["cols"]
        else:
            feat_cols = [col_name]

        # fill the nan values of only the columns that are in the fill_cols dict
        if col_name in self.fill_cols.keys():
            # the fill_cols dict contains the columns that need to be filled with a specific value
            df.loc[:, feat_cols] = df.loc[:, feat_cols].fillna(self.fill_cols[col_name])

        # remove the rows that have all the values in the columns to preprocess as nan
        df = df.dropna(subset=feat_cols, how="all").reset_index(drop=True)
        vvprint(
            f"Number of rows after removing null values in {col_name}: {df.shape[0]}"
        )

        # preprocess the column based on the column name and the type of the feature
        if col_name == "misc_price":
            df.loc[:, col_name] = DataTransformer.process_misc_price(
                df.loc[:, col_name], self.conversion_rates, verbose
            )
        elif col_name in self.float_features.keys():
            df.loc[:, col_name] = DataTransformer.process_float_feature(
                df.loc[:, col_name], self.float_features[col_name], verbose
            )
        elif col_name in self.binary_features.keys():
            pat = self.binary_features[col_name]["pat"]
            repl = self.binary_features[col_name]["repl"]
            new_col_name = self.binary_features[col_name]["new_col"]

            df.loc[:, col_name] = DataTransformer.process_binary_feature(
                df.loc[:, col_name], pat, repl, verbose
            )

            # rename the column to the new column name
            df = df.rename(columns={col_name: new_col_name})
            col_name = new_col_name
            feat_cols = [col_name]
        elif col_name in self.multi_col_features.keys():
            pat = self.multi_col_features[col_name]["pat"]
            new_cols = self.multi_col_features[col_name]["new_cols"]

            new_df = DataTransformer.process_multi_col_features(
                df.loc[:, col_name], pat, new_cols, verbose
            )

            # concat the new columns to the dataframe and drop the old column
            df = pd.concat([df, new_df], axis=1)
            df = df.drop(columns=feat_cols)

            # update the col_name and feat_cols to the new columns
            col_name = new_df.columns.to_list()
            feat_cols = col_name
        elif col_name in self.multi_features.keys():
            df.loc[:, col_name] = DataTransformer.process_multi_feature(
                df.loc[:, col_name], self.multi_features[col_name]
            )
        elif col_name in ["main_camera_cols", "selfie_camera_cols"]:
            camera_cols = self.camera_features[col_name]["cols"]
            pat = self.camera_features[col_name]["pat"]

            new_df = DataTransformer.process_camera_features(
                df.loc[:, feat_cols], col_name, camera_cols, pat, verbose
            )

            # update the df with the new columns and drop the old columns
            df = pd.concat([df, new_df], axis=1)
            df = df.drop(columns=feat_cols)

            # update the col_name and feat_cols to the new columns
            col_name = new_df.columns.to_list()
            feat_cols = col_name
        elif col_name in self.concat_features.keys():
            new_df = df[feat_cols].apply(
                lambda x: " - ".join(x.astype(str)), axis=1
            ).to_frame()
            new_df.columns = [col_name]

            df = pd.concat([df, new_df], axis=1)
            df = df.drop(columns=feat_cols)

            # update the col_name and feat_cols to the new columns
            col_name = new_df.columns.to_list()
            feat_cols = col_name
        else:
            vvprint(f"Column {col_name} not processed!")
            raise NotImplementedError

        # remove the rows that are nan after preprocessing
        df = df.dropna(subset=feat_cols, how="all").reset_index(drop=True)
        vvprint(f"Number of rows after extracting valid {col_name}: {df.shape[0]}\n")
        return df


def extract_prices_rate(curr, ser):
    # pattern to extract the price
    num_pat = r"[\d]{2,6}(?:\.[\d]{1,2})?"
    pattern = f"EUR\s?(?P<eur>{num_pat}).*?{curr}\s?(?P<{curr.lower()}>{num_pat})"

    # filter the rows that contain the currency of interest
    prices = ser[ser.str.contains(f"EUR.*?{curr}|{curr}.*?EUR", regex=True, na=False)]

    # extract the price values of both currencies and drop the rows with NaN values
    prices_ext = (
        prices.str.extract(pattern, flags=re.IGNORECASE).astype(float).dropna(how="any")
    )
    prices_ext = prices_ext.sort_values("eur")

    conversion_rate = prices_ext.apply(lambda x: x[1] / x[0], axis=1)
    conversion_rate = conversion_rate.rename("conversion_rate")

    print(f"Number of prices: {len(prices_ext)}")
    return prices_ext, conversion_rate
