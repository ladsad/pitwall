import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window


TIMEDELTA_COLS = [
    "LapTime",
    "Sector1Time",
    "Sector2Time",
    "Sector3Time",
    "PitInTime",
    "PitOutTime",
]

def timedeltas_to_seconds(pdf: pd.DataFrame) -> pd.DataFrame:
    
    for col in TIMEDELTA_COLS:
        if col in pdf.columns:
            pdf[col] = pd.to_timedelta(pdf[col], errors="coerce").dt.total_seconds()
    return pdf


def drop_critical_nulls(df: DataFrame) -> DataFrame:
    
    critical_cols = ["Driver", "Team", "LapTime", "session_type", "season", "event"]
    return df.dropna(subset=critical_cols)


def cast_lap_number(df: DataFrame) -> DataFrame:
    
    return df.withColumn("LapNumber", F.col("LapNumber").cast("integer"))


def normalise_compound(df: DataFrame) -> DataFrame:
    
    known = ["SOFT", "MEDIUM", "HARD", "INTER", "WET"]
    return df.withColumn(
        "Compound",
        F.when(F.upper(F.col("Compound")).isin(known), F.upper(F.col("Compound")))
         .otherwise(F.lit("UNKNOWN"))
    )


def filter_outlier_laps(df: DataFrame, threshold: float = 1.07) -> DataFrame:

    w = Window.partitionBy("season", "event", "session_type")
    df = df.withColumn("_session_best", F.min("LapTime").over(w))
    df = df.filter(F.col("LapTime") <= F.col("_session_best") * threshold)
    return df.drop("_session_best")