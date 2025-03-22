# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/manufacturing-root-cause-analysis/).

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Create a model serving endpoint for online causal analysis
# MAGIC
# MAGIC With our structural causal model registered in Unity Catalog, the final step is deploying it behind a Model Serving endpoint. This setup is crucial for conducting online causal analyses, such as anomaly attribution, in real time. Identifying the root causes of defective products quickly helps minimize the costs associated with their impact.
# MAGIC
# MAGIC This notebook demonstrates how to streamline Python-based model serving workflows. It uses [Databricks SDK](https://docs.databricks.com/en/dev-tools/sdk-python.html) for creating a model serving endpoint, updating the endpoint configuration to use specific model versions, making prediction requests, and deleting endpoints when needed.
# MAGIC
# MAGIC See the notebook `01_causal_graph` for a recommended cluster configuration.

# COMMAND ----------

# MAGIC %pip install dowhy==0.12 --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define variables and set MLflow experiment

# COMMAND ----------

import mlflow

# Set the registry URI to Databricks Unity Catalog
mlflow.set_registry_uri("databricks-uc")

# Create an MLflow client to interact with the tracking server
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

user_name = spark.sql("SELECT current_user()").collect()[0][0]
first_name = user_name.split(".")[0]

# Set up Unity Catalog
catalog = f'causal_solacc_{first_name}'     # Change this to your catalog name
schema = f'rca'                             # Change this to your schema name
model = f"manufacturing_rca"                # Change this to your model name
log_schema = "log"                          # A schema within the catalog where the inferece log is going to be stored 
model_name = f"{catalog}.{schema}.{model}"  # An existing model in model registry, may have multiple versions
model_serving_endpoint_name = f"{model}_{first_name}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Set up configurations
# MAGIC Based on your latency and throughput requirements, it’s important to select the appropriate `workload_type` and `workload_size`. The `auto_capture_config` block defines where to store inference logs, including the requests and responses from the endpoint, along with their timestamps.

# COMMAND ----------

# Get the champion model version
champion_version = client.get_model_version_by_alias(model_name, "champion")
model_version = champion_version.version

# Define the JSON configuration for the model serving endpoint
my_json = {
    "name": model_serving_endpoint_name,
    "config": {
        "served_models": [
            {
                "model_name": model_name,
                "model_version": model_version,
                "workload_type": "CPU",
                "workload_size": "Small",
                "scale_to_zero_enabled": "true",
            }
        ],
        "auto_capture_config": {
            "catalog_name": catalog,
            "schema_name": log_schema,
            "table_name_prefix": model_serving_endpoint_name,
        },
    },
}

# Ensure the schema for the inference table exists
_ = spark.sql(
    f"CREATE SCHEMA IF NOT EXISTS {catalog}.{log_schema}"
)

# Drop the inference table if it exists
_ = spark.sql(
    f"DROP TABLE IF EXISTS {catalog}.{log_schema}.`{model_serving_endpoint_name}_payload`"
)

# COMMAND ----------

# MAGIC %md
# MAGIC The following cell defines Python functions that:
# MAGIC - Create a model serving endpoint
# MAGIC - Update a model serving endpoint configuration with the latest model version
# MAGIC - Delete a model serving endpoint

# COMMAND ----------

import mlflow.deployments

def func_create_endpoint(json):
    client = mlflow.deployments.get_deploy_client("databricks")
    try:
        client.get_deployment(json["name"])
        new_model_version = json["config"]["served_models"][0]["model_version"]
        client.update_deployment(
            name=json["name"], 
            config=json["config"]
        )
    except:
        client.create_endpoint(
            name = model_serving_endpoint_name,
            config = json["config"],
            )

def func_delete_model_serving_endpoint(json):
    client = mlflow.deployments.get_deploy_client("databricks")
    client.delete_endpoint(json["name"])
    print(json["name"], "endpoint is deleted!")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's create an endpoint.

# COMMAND ----------

func_create_endpoint(my_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wait for the endpoint to be ready
# MAGIC
# MAGIC The `wait_for_endpoint()` function below, defined in the following command, retrieves and returns the status of the serving endpoint. We will wait until the endpoint is fully ready.

# COMMAND ----------

def wait_for_endpoint(endpoint_name):
    '''Wait for a model serving endpoint to be ready'''
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
    import time

    # Initialize WorkspaceClient
    w = WorkspaceClient()
    state = ""
    for i in range(200):
        state = w.serving_endpoints.get(endpoint_name).state
        if state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
            if i % 40 == 0:
                print(f"Waiting for endpoint to deploy {endpoint_name}. Current state: {state}")
            time.sleep(10)
        elif state.ready == EndpointStateReady.READY:
            print('endpoint ready.')
            return
        else:
            break
    raise Exception(f"Couldn't start the endpoint, timeout, please check your endpoint for more details: {state}")

wait_for_endpoint(my_json["name"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score the model
# MAGIC
# MAGIC Once the endpoint is operational, we can start sending requests. The following cell defines the `get_anomaly_attribution()` function, which sends an anomaly attribution request to the endpoint.

# COMMAND ----------

from mlflow.deployments import get_deploy_client

def get_anomaly_attribution(endpoint, dataset):
    client = get_deploy_client("databricks")
    ds_dict = {"dataframe_split": dataset.to_dict(orient="split")}
    response = client.predict(endpoint=endpoint, inputs=ds_dict)
    return response["predictions"][0]

# COMMAND ----------

# MAGIC %md
# MAGIC Let’s query the table containing the training data and select a sample with `quality = 1` to test the endpoint.

# COMMAND ----------

import pandas as pd

train = spark.read.table(f"{catalog}.{schema}.data_manufacturing")
train = train.toPandas()
defects = train[train['quality'] == 1]

display(pd.DataFrame([defects.iloc[0]]))

# COMMAND ----------

# MAGIC %md 
# MAGIC Send a request using Databricks SDK.

# COMMAND ----------

import pandas as pd
from mlflow.deployments import get_deploy_client

dataset = pd.DataFrame([defects.iloc[0]])
result = get_anomaly_attribution(my_json["name"], dataset)

display(result)

# COMMAND ----------

# MAGIC %md
# MAGIC Looks great! But we can even visualize the response:

# COMMAND ----------

import dowhy

# Plot the anomaly attribution scores
dowhy.utils.bar_plot({k: v for k, v in result.items()}, ylabel='Anomaly attribution score')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delete the endpoint
# MAGIC
# MAGIC Although the endpoint is configured to scale down to zero when there is no incoming traffic, let’s clean up and delete the endpoint:

# COMMAND ----------

func_delete_model_serving_endpoint(my_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap up
# MAGIC
# MAGIC That’s it! In this notebook, we took our structural causal model registered in Unity Catalog and deployed it behind a Model Serving endpoint. Additionally, we explored how to interact with this endpoint using the Databricks SDK.

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
