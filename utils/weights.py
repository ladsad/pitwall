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
    
    sessions = list(SESSION_WEIGHTS.items())
    weights_col = F.when(F.col("session_type") == sessions[0][0], F.lit(sessions[0][1]))
    for session, weight in sessions[1:]:
        weights_col = weights_col.when(F.col("session_type") == session, F.lit(weight))
    weights_col = weights_col.otherwise(F.lit(None).cast("float"))

    return df.withColumn("session_weight", weights_col)


def add_sample_weight(df: DataFrame) -> DataFrame:
    
    df = add_recency_weight(df)
    df = add_session_weight(df)
    return df.withColumn(
        "sample_weight",
        F.col("recency_weight") * F.col("session_weight")
    )