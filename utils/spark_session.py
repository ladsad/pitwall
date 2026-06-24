import os
import sys
from pyspark.sql import SparkSession

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# Force PySpark to use the Java runtime bundled in our conda env (Windows only)
if os.name == "nt":
    conda_dir = os.path.dirname(sys.executable)
    os.environ["JAVA_HOME"] = os.path.join(conda_dir, "Library")

    # Set HADOOP_HOME for Windows compatibility (winutils.exe)
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hadoop_home = os.path.join(project_dir, "hadoop")
    os.environ["HADOOP_HOME"] = hadoop_home

    # CRITICAL: add hadoop/bin to PATH so hadoop.dll can be loaded to fix NativeIO access0 error
    hadoop_bin = os.path.join(hadoop_home, "bin")
    os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

def _is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ

def get_spark_session(app_name: str = "pitwall") -> SparkSession:
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark
