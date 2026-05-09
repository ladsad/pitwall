from pyspark.sql import DataFrame
from pyspark.sql import functions as F

LAMBDA = 0.15

SESSION_WEIGHTS = {
    "R":   1.0,
    "Q":   0.7,
    "SQ":  0.6,
    "S":   0.5,
    "FP3": 0.35,
    "FP2": 0.25,
    "FP1": 0.15,
}

def add_recency_weight(df: DataFrame) -> DataFrame:
    max_round = df.agg(F.max("round_number")).collect()[0][0]
    return df.withColumn(
        "recency_weight",
        F.exp(-LAMBDA * (F.lit(max_round) - F.col("round_number")))
    )

def add_session_weight(df: DataFrame) -> DataFrame:
    weights_col = None
    for session, weight in SESSION_WEIGHTS.items():
        if weights_col is None:
            weights_col = F.when(F.col("session_type") == session, F.lit(weight))
        else:
            weights_col = weights_col.when(F.col("session_type") == session, F.lit(weight))
        
    return df.withColumn("session_weight", weights_col)

def add_sample_weight(df: DataFrame) -> DataFrame:
    df = add_recency_weight(df)
    df = add_session_weight(df)
    return df.withColumn(
        "sample_weight",
        F.col("recency_weight") * F.col("session_weight")
    )