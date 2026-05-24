"""
utils/tel_schema.py
────────────────────
Explicit schemas for the telemetry Bronze and Silver layers.
Never use inferSchema=True — per project convention.
"""

from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, FloatType, BooleanType, ArrayType,
)

# ── BRONZE SCHEMA ─────────────────────────────────────────────────────────────
# One row per lap. Channel columns are variable-length arrays (raw 200 Hz data).
# Written by 07_tel_ingest.py.

TEL_BRONZE_SCHEMA = StructType([
    StructField("lap_id",       StringType(),                    nullable=False),  # {year}_{gp}_{driver}_{lap_n}
    StructField("driver",       StringType(),                    nullable=False),  # 3-letter code
    StructField("season",       IntegerType(),                   nullable=False),
    StructField("event",        StringType(),                    nullable=False),
    StructField("session_type", StringType(),                    nullable=False),  # FP1/FP2/.../R
    StructField("lap_number",   IntegerType(),                   nullable=True),
    StructField("distance",     ArrayType(FloatType()),          nullable=True),   # metres, raw
    StructField("speed",        ArrayType(FloatType()),          nullable=True),   # km/h
    StructField("throttle",     ArrayType(FloatType()),          nullable=True),   # 0–100
    StructField("brake",        ArrayType(BooleanType()),        nullable=True),
    StructField("rpm",          ArrayType(FloatType()),          nullable=True),
    StructField("gear",         ArrayType(IntegerType()),        nullable=True),
    StructField("drs",          ArrayType(IntegerType()),        nullable=True),   # 0/8/10/12/14
])

# ── SILVER SCHEMA ─────────────────────────────────────────────────────────────
# One row per lap. `channels` holds a resampled, normalised (6, 1024) tensor
# stored as a list-of-lists: outer dim = 6 channels, inner dim = 1024 values.
# Written by 08_tel_preprocess.py.

TEL_SILVER_SCHEMA = StructType([
    StructField("lap_id",        StringType(),                               nullable=False),
    StructField("driver",        StringType(),                               nullable=False),
    StructField("season",        IntegerType(),                              nullable=False),
    StructField("event",         StringType(),                               nullable=False),
    StructField("session_type",  StringType(),                               nullable=False),
    StructField("lap_number",    IntegerType(),                              nullable=True),
    StructField("channels",      ArrayType(ArrayType(FloatType())),          nullable=False),  # (6, 1024)
    StructField("label_position",IntegerType(),                              nullable=True),   # race finishing pos; null mid-weekend
])
