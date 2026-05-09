from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from config import SESSION_WEIGHTS, RECENCY_LAMBDA

def add_recency_weight(df: DataFrame) -> DataFrame:
    max_round = df.agg(F.max("round_number")).collect()[0][0]
    return df.withColumn(
        "recency_weight",
        F.exp(-RECENCY_LAMBDA * (F.lit(max_round) - F.col("round_number")))
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