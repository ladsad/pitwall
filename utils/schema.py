from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, BooleanType
)

BRONZE_SCHEMA = StructType([
    StructField("Driver",         StringType(),  nullable=True),
    StructField("DriverNumber",   StringType(),  nullable=True),
    StructField("Team",           StringType(),  nullable=True),
    StructField("LapNumber",      IntegerType(), nullable=True),
    StructField("LapTime",        FloatType(),   nullable=True),
    StructField("Sector1Time",    FloatType(),   nullable=True),
    StructField("Sector2Time",    FloatType(),   nullable=True),
    StructField("Sector3Time",    FloatType(),   nullable=True),
    StructField("SpeedI1",        FloatType(),   nullable=True),
    StructField("SpeedI2",        FloatType(),   nullable=True),
    StructField("SpeedFL",        FloatType(),   nullable=True),
    StructField("SpeedST",        FloatType(),   nullable=True),
    StructField("Compound",       StringType(),  nullable=True),
    StructField("TyreLife",       FloatType(),   nullable=True),
    StructField("FreshTyre",      StringType(),  nullable=True),  # FastF1 sends as string
    StructField("PitInTime",      FloatType(),   nullable=True),
    StructField("PitOutTime",     FloatType(),   nullable=True),
    StructField("IsPersonalBest", StringType(),  nullable=True),  # same — check before casting
    StructField("session_type",   StringType(),  nullable=False), # always tagged on ingest
    StructField("season",         IntegerType(), nullable=False),
    StructField("event",          StringType(),  nullable=False),
    StructField("session",        StringType(),  nullable=False),
])

GOLD_SCHEMA = StructType([
    StructField("driver",            StringType(),  nullable=False),
    StructField("team",              StringType(),  nullable=False),
    StructField("round_number",      IntegerType(), nullable=False),
    StructField("event",             StringType(),  nullable=False),
    StructField("season",            IntegerType(), nullable=False),
    StructField("session_type",      StringType(),  nullable=False),
    StructField("lap_time_delta",    FloatType(),   nullable=True),
    StructField("consistency_score", FloatType(),   nullable=True),
    StructField("best_sector_combo", FloatType(),   nullable=True),
    StructField("tyre_deg_rate",     FloatType(),   nullable=True),
    StructField("pace_vs_teammate",  FloatType(),   nullable=True),
    StructField("pace_trend",        FloatType(),   nullable=True),
    StructField("race_position",     IntegerType(), nullable=True),  # null mid-weekend
    StructField("recency_weight",    FloatType(),   nullable=True),
    StructField("session_weight",    FloatType(),   nullable=True),
    StructField("sample_weight",     FloatType(),   nullable=True),
])