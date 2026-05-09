import os
from pyspark.sql import SparkSession

def _is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

def get_spark_session(app_name: str = "pitwall") -> SparkSession:
    if _is_databricks():
        return SparkSession.builder.getOrCreate()
    
    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    return spark
