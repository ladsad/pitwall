from utils.spark_session import get_spark_session
import os

print("Testing Spark Session...")
try:
    spark = get_spark_session("test")
    print("Spark Session created successfully.")
    print(f"Spark Version: {spark.version}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
