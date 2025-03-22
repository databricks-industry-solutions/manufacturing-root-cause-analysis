# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/manufacturing-root-cause-analysis/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Causal Relationships
# MAGIC
# MAGIC In this first notebook, we will:
# MAGIC
# MAGIC 1. Understand the use case.
# MAGIC 2. Generate a synthetic dataset.
# MAGIC 3. Construct a causal graph.
# MAGIC 4. Log the graph to MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cluster configuration
# MAGIC We recommend using a cluster with the following or similar specifications to run this solution accelerator:
# MAGIC - Unity Catalog enabled cluster
# MAGIC - Databricks Runtime 15.4 LTS ML or above
# MAGIC - Single-node cluster: e.g. `m5d.2xlarge` on AWS or `Standard_D8ds_v5` on Azure Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies

# COMMAND ----------

# DBTITLE 1,Install graphviz from nicer visualization
# MAGIC %sh apt-get update && apt-get install -y graphviz graphviz-dev

# COMMAND ----------

# MAGIC %md
# MAGIC We install the required libraries from the `requirements.txt`.

# COMMAND ----------

# DBTITLE 1,Install requirements
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
catalog = f'causal_solacc_{first_name}'     # Change this to your catalog name
schema = f'rca'                             # Change this to your schema name

setup_unity_catalog(catalog, schema)

# COMMAND ----------

# Set the experiment name
experiment_name = f"/Users/{user_name}/rca_manufacturing"
mlflow.set_experiment(experiment_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use Case
# MAGIC
# MAGIC Our goal with causal AI is to identify the true root causes behind a drop in quality, whether in a specific product or a batch of products. This enables us to implement effective measures to counteract these quality deviations, prevent recurrence, minimize waste, and improve overall product quality. In the notebooks, we will explore how to achieve this for a specific manufacturing process shown below. 
# MAGIC
# MAGIC The process flow shows how different factors influence product quality:
# MAGIC
# MAGIC 1. **Input Factors**:
# MAGIC    - Worker (Manual Operator)
# MAGIC    - Machine Settings
# MAGIC    - Material Properties
# MAGIC    - Environment* (Temperature, Pressure, Humidity in the Chamber)
# MAGIC
# MAGIC 2. **Process Measurements**:
# MAGIC    - Position & Alignment
# MAGIC    - Force & Torque
# MAGIC    - Temperature
# MAGIC
# MAGIC 3. **Quality Checks**:
# MAGIC    - Dimensions
# MAGIC    - Torque Checks
# MAGIC    - Visual Inspection
# MAGIC
# MAGIC These factors combine to determine the final quality outcome. When quality drops unexpectedly, we'll use DoWhy to trace the root cause through these causal relationships.

# COMMAND ----------

from IPython.display import Image, display
display(Image('./images/manufacturing-process-A-simplified.png', width=1000))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Example Cause-Effect Relationships
# MAGIC
# MAGIC For this manufacturing example, domain experts identified the following relationships. Note that these relationships are specific to this use case - other manufacturing processes or domains may have different causal relationships that can be analyzed using the same methodology.
# MAGIC
# MAGIC **Process Inputs → Measurements**
# MAGIC - Worker & Machine → Position & Alignment
# MAGIC   - Worker skill level and experience affects positioning precision
# MAGIC   - Machine settings influence alignment accuracy
# MAGIC
# MAGIC - Raw Material & Material → Force & Torque
# MAGIC   - Material properties from different suppliers require varying processing forces
# MAGIC   - Raw material characteristics affect required torque levels
# MAGIC
# MAGIC - Environment → Temperature
# MAGIC   - Chamber conditions (temperature, humidity, pressure) affect interface temperature
# MAGIC   - Higher humidity may cause cooling through condensation
# MAGIC
# MAGIC **Measurements → Quality Checks**
# MAGIC - Position & Alignment → Dimensions
# MAGIC   - Misalignment leads to dimensional failures
# MAGIC
# MAGIC - Force & Torque → Dimensions & Torque
# MAGIC   - Excessive force may cause dimensional issues
# MAGIC   - Insufficient force causes weak joints (fails torque check)
# MAGIC   
# MAGIC - Temperature → Visual & Torque
# MAGIC   - High temperatures can cause visible defects
# MAGIC   - Low temperatures may result in weak bonds
# MAGIC
# MAGIC **Final Quality**
# MAGIC - Any failed check (Dimensions, Torque, Visual) results in overall quality failure
# MAGIC
# MAGIC This causal graph structure allows us to trace quality issues back to their root causes. The same methodology can be applied to other processes by adapting the variables and relationships to the specific context.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate data
# MAGIC
# MAGIC Because this is a fictitious use case, we generate a synthetic dataset that aligns with our causal graph. Performing causal analysis on a synthetic dataset allows us to validate the approach and better understand the technique. After generating the dataset, we store it in a Delta table for later use. For more details, refer to the `generate_data` function in the `99_utils` notebook.

# COMMAND ----------

X = generate_data(catalog, schema, 1000)

display(X)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate a causal graph
# MAGIC
# MAGIC From the relationships identified with our domain experts, we will construct our causal graph using the `DiGraph` class from the `networkx` package. Here the package `graphviz` and `pygraphviz` give us a nicely formatted DAG show below. 

# COMMAND ----------

true_graph = nx.DiGraph()
true_graph.add_nodes_from = X.columns

dependencies = {
        "position_alignment": ['worker', 'machine'],
        "force_torque": ['raw_material', 'machine', 'material'],
        "temperature": ['chamber_temperature', 'chamber_humidity', 'chamber_pressure'],
        "dimensions": ['position_alignment', 'force_torque'],
        "torque_checks": ['force_torque', 'temperature'],
        "visual_inspection": ['temperature'],
        "quality": ['dimensions', 'torque_checks', 'visual_inspection'],
    }

for child in dependencies:
    for parent in dependencies[child]:
        true_graph.add_edge(parent, child)

dowhy.gcm.util.plot(true_graph, figure_size=(20, 20))

# COMMAND ----------

# MAGIC %md
# MAGIC We established the causal relationships between our variables in collaboration with domain experts. However, this process is not always straightforward, as scheduling time with experts can be costly, and even they may not have full knowledge of all the details. In such situations, we can turn to automated causal discovery algorithms. While these algorithms typically do not produce a perfect graph, they can serve as a valuable starting point. For more details, refer to the section `Appendix A` in the notebook `05_appendix`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the causal graph to MLflow
# MAGIC
# MAGIC Because this graph will be used in subsequent notebooks, we will log it as an artifact using `MLflow`.

# COMMAND ----------

with mlflow.start_run(run_name="causal_graph") as run:
    
    # save graph object to file
    pickle.dump(true_graph, open('/databricks/driver/causal_graph.pickle', 'wb'))

    # log the pickle file to mlflow
    mlflow.log_artifact("/databricks/driver/causal_graph.pickle", artifact_path="graph")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap up
# MAGIC
# MAGIC In this notebook, we explored a manufacturing production line as an example use case. We generated a synthetic dataset, created a causal graph, and logged the graph using MLflow. These steps reflect the initial stages of a causal AI project for root cause analysis: understanding the use case, collecting data, gathering insights from domain experts and establishing causal relationships while logging all created artifacts along the way.
# MAGIC
# MAGIC In the next notebook, `02_causal_modeling`, we will delve into integrating the causal graph with observational data (in our case, the synthetic dataset).

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
