import os
import sys
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

def _is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

def get_spark_session(app_name: str = "pitwall") -> SparkSession:
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark
