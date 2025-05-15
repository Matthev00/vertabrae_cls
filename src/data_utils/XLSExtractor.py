import re
from pathlib import Path

import pandas as pd


class XLSExtractor:
    """
    A class to extract data from XLS files.
    """

    def __init__(self, file_path: Path):
        """
        Initializes the XLSExtractor with the path to the XLS file.

        :param file_path: Path to the XLS file.
        """
        self.file_path = file_path
        self.data = None
        self._load_data()
        self.records: list[tuple[str, str]] = []

    def _load_data(self) -> None:
        """
        Loads data from the XLS file into a pandas DataFrame.
        """
        try:
            self.data = pd.read_excel(self.file_path)
        except Exception as e:
            raise ValueError(f"Failed to load XLS file: {e}")

    def _drop_irrelevant_columns(self) -> None:
        """
        Drops irrelevant columns from the DataFrame.
        """
        columns_of_interest = [
            "I.I.",
            "Poziom",
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
            "B1",
            "B2",
            "B3",
            "C",
        ]
        self.data = self.data[columns_of_interest].copy()

    def _extract_records(self) -> None:
        """
        Extracts records from the DataFrame and stores them in the records list.
        """
        current_traumas = []
        current_ii = None

        for _, row in self.data.iterrows():
            if pd.notna(row["I.I."]):
                if current_traumas:
                    self.records.append({"I.I": f"{current_ii}", "traumas": current_traumas})
                current_traumas = []
                current_ii = row["I.I."]

            if pd.notna(row["Poziom"]):
                levels = re.split(r"[/\-]", row["Poziom"])
                for level in levels:
                    level = level.strip()
                    for trauma_type in ["A0", "A1", "A2", "A3", "A4", "B1", "B2", "B3", "C"]:
                        if pd.notna(row[trauma_type]):
                            current_traumas.append((level, trauma_type))

        if current_traumas:
            self.records.append({"I.I": f"{current_ii}", "traumas": current_traumas})

    def extract_and_save(self, output_path: Path) -> None:
        """
        Extracts records from the XLS file and saves them to a CSV file.

        :param output_path: Path to save the CSV file.
        """
        self._drop_irrelevant_columns()
        self._extract_records()

        df = pd.DataFrame(self.records)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR

    extractor = XLSExtractor(RAW_DATA_DIR / "raport.xlsx")
    extractor.extract_and_save(INTERIM_DATA_DIR / "extracted_data.csv")
