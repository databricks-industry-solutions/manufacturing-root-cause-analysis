# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/manufacturing-root-cause-analysis/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Fit Causal Models to Data
# MAGIC
# MAGIC In this second notebook, we will:
# MAGIC
# MAGIC 1. Assign causal mechanisms to the causal graph defined in the previous notebook.
# MAGIC 2. Fit the causal models identified in the previous step to the causal graph.
# MAGIC 3. Evaluate the fitted graph to assess how well it represents the underlying data generation process.
# MAGIC 4. Register the fitted graph to Unity Catalog using MLflow for future use.
# MAGIC
# MAGIC See the notebook `01_causal_graph` for a recommended cluster configuration.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install Dependencies

# COMMAND ----------

# DBTITLE 1,Install graphviz from nicer visualization
# MAGIC %sh apt-get update && apt-get install -y graphviz graphviz-dev

# COMMAND ----------

# MAGIC %pip install -r ./requirements.txt --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC In the next cell, we run the `99_utils` notebook, which defines a few utility functions that we will use along the way.

# COMMAND ----------

# DBTITLE 1,Run utils notebook
# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define variables and set MLflow experiment

# COMMAND ----------

import mlflow
import pickle
import numpy as np
import pandas as pd
import dowhy
import networkx as nx

# COMMAND ----------

user_name = spark.sql("SELECT current_user()").collect()[0][0]
first_name = user_name.split(".")[0]

# Set up Unity Catalog
catalog = f'causal_solacc_{first_name}'     # Change this to your catalog name
schema = f'rca'                             # Change this to your schema name
model = f"manufacturing_rca"                # Change this to your model name

setup_unity_catalog(catalog, schema)

# COMMAND ----------

# Set the experiment name
experiment_name = f"/Users/{user_name}/rca_manufacturing"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the causal graph
# MAGIC
# MAGIC Now, let's load the causal graph defined in the previous notebook. We will integrate this graph with generative models that describe the data generation process at each node to construct a structural causal model (SCM).
# MAGIC
# MAGIC The causal graph can be loaded using MLflow:

# COMMAND ----------

# Find all the runs from the prior notebook for causal discovery
client = mlflow.MlflowClient()
experiment = mlflow.get_experiment_by_name(experiment_name)
discovery_runs = client.search_runs(
    experiment_ids=[experiment.experiment_id], 
    filter_string="attributes.run_name='causal_graph'",
    order_by=["start_time DESC"],
    max_results=1,
    )

# Make sure there is at least one run available
assert len(discovery_runs) == 1, "Run the previous notebook: 01_causal_graph"

# The only result should be the latest based on our search_runs call
latest_discovery_run = discovery_runs[0]
latest_discovery_run.info.artifact_uri

# Load the graph artifact from the run
local_path = mlflow.artifacts.download_artifacts(latest_discovery_run.info.artifact_uri + "/graph/causal_graph.pickle")

with open(local_path, "rb") as f:
    causal_graph = pickle.load(f)

# COMMAND ----------

# MAGIC %md
# MAGIC To verify, let's plot the graph:

# COMMAND ----------

dowhy.gcm.util.plot(causal_graph, figure_size=(20, 20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the dataset
# MAGIC
# MAGIC Next, we will load the synthetic dataset we generated in the previous notebook.

# COMMAND ----------

# Define the Delta table name
table_name = f"{catalog}.{schema}.data_manufacturing"

# Query to retrieve the version history of the Delta table
version_query = f"DESCRIBE HISTORY {table_name}"

# Get the latest version of the Delta table
version = spark.sql(version_query).collect()[0][0]

# Read the Delta table as of the latest version
sdf = spark.read.format("delta").option("versionAsOf", version).table(table_name)

# Convert the Spark DataFrame to a pandas DataFrame
pdf = sdf.toPandas()

# Display the first few rows of the pandas DataFrame
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assign causal mechanisms to the graph
# MAGIC
# MAGIC As we can see, we have one sample for each processed product, including all the variables in the causal graph.
# MAGIC
# MAGIC While we have defined the causal graph, we still need to assign generative models to its nodes. These models can either be manually specified and configured if necessary or automatically inferred from the data using heuristics. Here, we will use the latter approach:

# COMMAND ----------

from dowhy import gcm
np.random.seed(1)

# Create the structural causal model object
scm = gcm.StructuralCausalModel(causal_graph)

# Automatically assign generative models to each node based on the given data
auto_assignment_summary = gcm.auto.assign_causal_mechanisms(
  scm, 
  pdf, 
  override_models=True, 
  quality=gcm.auto.AssignmentQuality.GOOD
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Whenever possible, it is best practice to assign models based on prior knowledge, as this ensures they closely reflect the underlying physics of the domain rather than relying on data-specific nuances. However, in this case, we have asked DoWhy to handle this task for us.
# MAGIC
# MAGIC Once the models are automatically assigned, we can print a summary to gain insights into the selected models:

# COMMAND ----------

print(auto_assignment_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC The auto-assignment function evaluates both linear and non-linear models for non-root nodes, considering Additive Noise Models (ANMs) for continuous data (e.g., position_alignment) and Discrete ANMs for discrete data (e.g., dimensions), selecting the best-performing model based on metrics such as MSE.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit causal models to the data
# MAGIC
# MAGIC After assigning a model to each node, we need to learn the parameters of the model. We fit the causal model to the dataset.

# COMMAND ----------

gcm.fit(scm, pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the fitted causal models
# MAGIC
# MAGIC The fit method trains the generative models for each node by learning their parameters. Let's examine the performance of these causal mechanisms and evaluate how well they capture the underlying distribution:

# COMMAND ----------

print(
  gcm.evaluate_causal_model(
  scm,
  pdf, 
  compare_mechanism_baselines=True, 
  evaluate_invertibility_assumptions=True)
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Broadly, the `gcm.evaluate_causal_model` method performs four types of evaluations on the fitted graph: evaluation of causal mechanisms, assessment of the invertible functional causal model assumption, evaluation of the generated distribution, and analysis of the causal graph structure. While we won't delve into the details of these tests here, we encourage users to check DoWhy's [documentation](https://www.pywhy.org/dowhy/v0.11.1/user_guide/modeling_gcm/model_evaluation.html) and [source code](https://github.com/py-why/dowhy/blob/main/dowhy/gcm/model_evaluation.py) for a deeper understanding.
# MAGIC
# MAGIC In our case, using a synthetically generated dataset, the fitted causal mechanisms largely align well with the data generation process. The above graph provides strong evidence that the causal graph structure identified in the model is capturing real and meaningful relationships in the data, rather than random associations. The extremely low p-values and clear separation between the original and permuted graphs suggest that the causal graph has successfully identified genuine structural relationships in the system being studied. 
# MAGIC
# MAGIC However, in real-world scenarios, datasets are often messier, have smaller sample sizes, or exhibit lower signal-to-noise ratios. In addition, the graph might be missing key confounders. For these reasons, it’s crucial to understand the evaluation techniques mentioned above and recognize how each test addresses specific issues.
# MAGIC
# MAGIC If the evaluation results indicate signs of misspecification, you can choose to revisit steps such as data collection, causal discovery, or modeling of causal mechanisms, or proceed with your analysis despite the issues. The evaluations provide insights into the quality of the causal model but they should not be overinterpreted, as some causal relationships are inherently challenging to model. Additionally, many algorithms demonstrate robustness to misspecifications or suboptimal performance of causal mechanisms.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the fitted causal graph to Unity Catalog using MLflow
# MAGIC
# MAGIC Once we are satisfied with our causal models, we can register it with Unity Catalog to ensure proper governance. Later, we will load this model to perform causal analysis. MLflow does not natively support the `gcm.StructuralCausalModel` (SCM) object, but we can simply wrap the SCM object using `mlflow.pyfunc.PythonModel` and log it with MLflow instead.

# COMMAND ----------

import mlflow

class SCM(mlflow.pyfunc.PythonModel):
  def __init__(self, scm, causal_graph, target_node):
    from dowhy import gcm
    import pandas as pd
    self.scm = scm
    self.causal_graph = causal_graph
    self.target_node = target_node

  def load_scm(self):
    return self.scm
  
  def load_causal_graph(self):
    return self.causal_graph
  
  def predict(self, context, input_df):
    return pd.DataFrame(gcm.attribute_anomalies(self.scm, target_node=self.target_node, anomaly_samples=input_df))

# COMMAND ----------

from dowhy import gcm
import sklearn
import mlflow
import mlflow.data
from mlflow.data.spark_dataset import SparkDataset
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec
from mlflow.models import infer_signature

# Define the input example for the model and infer its input-output signature
input_example = pdf.iloc[[0]]  # Select a single row as an input example
signature = infer_signature(
    model_input=input_example, 
    model_output=pd.DataFrame(gcm.attribute_anomalies(scm, target_node="quality", anomaly_samples=input_example)),
)

# Set the registered model name based on catalog, schema, and model
registered_model_name = f"{catalog}_{schema}_{model}"

# Start an MLflow run to log the causal model and its related metadata
with mlflow.start_run(run_name="causal_model") as run:
    # Log the causal model using MLflow's pyfunc interface
    mlflow.pyfunc.log_model(
        "model",
        python_model=SCM(scm, causal_graph, "quality"),  # Wrap the SCM object in a custom Python model
        pip_requirements=[
            "dowhy==" + dowhy.__version__,  # Log required package versions
            "pandas==" + pd.__version__,
            "numpy==" + np.__version__,
            "scikit-learn==" + sklearn.__version__,
        ],
        signature=signature,  # Log the inferred input-output signature
        input_example=input_example,  # Log an example input
        registered_model_name=registered_model_name,  # Register the model in Unity Catalog
    )
    
    # Log parameters related to the model's configuration or settings
    mlflow.log_params({
        **{
            "override_models": "True",  # Specify if existing models should be overridden
            "quality": "gcm.auto.AssignmentQuality.GOOD",  # Record the quality of assignments
        }
    })
    
    # Log the causal graph artifact for reference or reuse
    mlflow.log_artifact(local_path, artifact_path="causal_graph")
    
    # Log the input dataset used during the training or analysis process
    mlflow.log_input(
        mlflow.data.from_spark(df=sdf, table_name=table_name, version=version),  # Input dataset information
        context="training",  # Context of the dataset usage
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Let's assign the "champion" alias to the newly registered model. This makes it easier to load this specific version later.

# COMMAND ----------

from mlflow import MlflowClient
mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

# Assign an alias to the latest model version
def get_latest_model_version(mlflow_client, registered_name):
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{registered_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


model_version = get_latest_model_version(mlflow_client, registered_model_name)
mlflow_client.set_registered_model_alias(registered_model_name, "champion", model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap up
# MAGIC
# MAGIC This concludes the second notebook. Here, we assigned causal mechanisms to the causal graph defined in the previous notebook. We then fitted the graph on the dataset and evaluated the fitted graph to assess how well it captures the underlying data generation process. Finally, we registered the fitted graph in Unity Catalog using MLflow for future use. 
# MAGIC
# MAGIC In the next notebook, we will leverage the fitted graph to conduct causal analyses.

# COMMAND ----------

# MAGIC %md
# MAGIC © 2025 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Graphviz | An open source graph visualization software | Common Public License Version 1.0 | https://graphviz.org/download/
# MAGIC | pygraphviz | A Python interface to the Graphviz graph layout and visualization package | BSD | https://pypi.org/project/pygraphviz/
# MAGIC | networkx | A Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. | BSD | https://pypi.org/project/networkx/
# MAGIC | dowhy | A Python library for causal inference that supports explicit modeling and testing of causal assumptions | MIT | https://pypi.org/project/dowhy/
# MAGIC | causal-learn | A python package for causal discovery that implements both classical and state-of-the-art causal discovery algorithms, which is a Python translation and extension of Tetrad. | MIT | https://pypi.org/project/causal-learn/
# MAGIC | lime | Local Interpretable Model-Agnostic Explanations for machine learning classifiers | BSD | https://pypi.org/project/lime/
# MAGIC | shap | A unified approach to explain the output of any machine learning model | MIT | https://pypi.org/project/shap/
