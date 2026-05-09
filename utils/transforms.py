from pyspark.sql import DataFrame

# PySpark cannot serialise pandas Timedelta objects. Convert every timedelta
# column to total seconds (float) before handing the DataFrame to Spark.

TIMEDELTA_COLS = [
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "PitInTime",
    "PitOutTime",
]

def timedeltas_to_seconds(df: DataFrame) -> DataFrame:
    for col in TIMEDELTA_COLS:
        if col in df.columns and df[col].dtype == 'timedelta64[ns]':
            df[col] = df[col].dt.total_seconds().astype(float)
    return df
