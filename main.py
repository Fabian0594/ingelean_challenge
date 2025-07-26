import os
from dotenv import load_dotenv
from process_data import DataProcessor
from pandas_profiling import ProfileReport  # Asegúrate de tener instalado pandas-profiling


def main():
    load_dotenv()
    processor = DataProcessor()
    rute = os.getenv("rute")
    df = processor.load_data(rute)
    profile = ProfileReport(df, title="Reporte de Perfilado", explorative=True)
    profile.to_file("reporte_perfilado.html")


if __name__ == "__main__":
    main()