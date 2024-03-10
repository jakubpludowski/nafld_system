from nafld.utils.initialize_environment import initialize_environment
from nafld.utils.io_utils.load_csv_to_parquet import convert_csv_to_parquet

if __name__ == "__main__":
    CONF = initialize_environment()
    convert_csv_to_parquet(CONF.ORIGINAL_DATA_CSV_FILE, CONF.RAW_FEATURES_TABLE_PARQUET)
