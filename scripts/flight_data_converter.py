"""Convert flight data from xlsx to CSV (much faster to work with CSVs)."""
import pandas as pd


def convert_xlsx_to_csv(xlsx_file, csv_file):
    # Read the Excel file
    df = pd.read_excel(xlsx_file, skiprows=5, header=0)

    # Save the DataFrame to CSV
    df.to_csv(csv_file, index=False)


if __name__ == "__main__":
    convert_xlsx_to_csv(
        "bayes_air/data/wn_dec18_jan1.xlsx", "bayes_air/data/wn_dec18_jan1.csv"
    )
