# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/manufacturing-root-cause-analysis/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Appendix
# MAGIC
# MAGIC In this notebook, we explore two additional topics that are not immediately required for using the solution accelerator. The first focuses on automated or semi-automated causal discovery, and the second examines correlational machine learning and its explainability of causal factors.
# MAGIC
# MAGIC See the notebook `01_causal_graph` for a recommended cluster configuration.

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
# MAGIC %pip install lime --quiet
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

import numpy as np
import pandas as pd
import networkx as nx

# COMMAND ----------

user_name = spark.sql("SELECT current_user()").collect()[0][0]
first_name = user_name.split(".")[0]

# Set up Unity Catalog
catalog = f'causal_solacc_{first_name}'     # Change this to your catalog name
schema = f'rca'                             # Change this to your schema name

setup_unity_catalog(catalog, schema)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Appendix A: Causal Discovery
# MAGIC
# MAGIC Automated causal discovery algorithms aim to uncover causal relationships from observational data without requiring pre-defined structures. These algorithms use statistical dependencies, graph theory, and domain knowledge to infer causal links, making them powerful for analyzing complex systems with minimal prior assumptions. While they are efficient and scalable, their accuracy depends heavily on the quality of the data and assumptions like causal sufficiency and faithfulness, which may not always hold. Additionally, results often require validation by domain experts to ensure interpretability and reliability.
# MAGIC
# MAGIC [`causal-learn`](https://causal-learn.readthedocs.io/en/latest/index.html) is an open-source library that provides implementations of various causal discovery algorithms. Below is an example demonstrating the use of the PC algorithm.

# COMMAND ----------

import pandas as pd
from causallearn.search.ConstraintBased.PC import pc

data = spark.read.table(f"{catalog}.{schema}.data_manufacturing")
data = data.toPandas()

# default parameters
data = data.copy().drop('id', axis=1)
cg = pc(np.vstack(data.to_numpy()), node_names=data.columns)

# visualization using pydot
cg.draw_pydot_graph()

# COMMAND ----------

# MAGIC %md
# MAGIC The PC algorithm successfully captures some relationships, such as `machine` and `worker` influencing `position_alignment`, but it is far from perfect. Nonetheless, this provides a starting point for gathering feedback from domain experts to refine and improve the graph.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Appendix B: Correlation Machine Learning
# MAGIC
# MAGIC The need for causal machine learning in root cause analysis stems from the limitation of correlational machine learning, which is insufficient to identify the true source of variation in a target variable. Achieving this requires understanding and encoding causal relationships between attributes using a causal graph.
# MAGIC
# MAGIC In this section, we demonstrate how two widely adopted techniques—[LIME](https://github.com/marcotcr/lime) and [SHAP](https://shap.readthedocs.io/en/latest/)—when applied to a correlational model can result in ambiguous or even misleading attributions of anomalies. 
# MAGIC
# MAGIC We start by training an [XGBoost](https://xgboost.readthedocs.io/en/stable/) model for classification on our synthetically generated dataset. 

# COMMAND ----------

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, train_test_split

# Prepare data
X = data.drop('quality', axis=1)
y = data['quality']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = XGBClassifier()

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)

# Train the model
model.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's select only the defective samples from the training dataset.

# COMMAND ----------

samples = X_test[y_test == 1]

display(samples)

# COMMAND ----------

# MAGIC %md
# MAGIC We will first try `lime`:

# COMMAND ----------

from lime.lime_tabular import LimeTabularExplainer

# Use LIME to explain a prediction for a sample with quality = 1
explainer = LimeTabularExplainer(
  X_train.values, 
  feature_names=X_train.columns, 
  class_names=['quality'], 
  discretize_continuous=True
  )

exp = explainer.explain_instance(samples.iloc[0].values, model.predict_proba, num_features=len(X_train.columns))

# Display LIME explanation
exp.show_in_notebook()

# COMMAND ----------

# MAGIC %md
# MAGIC While `lime` accurately identifies attributes that contributed positively to the sample being classified as an anomaly, its attribution of root cause contribution is not informative. It assigns the highest weights to the variables `torque_checks`, `dimensions`, and `visual_inspections`, which are actually symptoms of the true causes. In this sample, we observe a failed `torque_checks`, and from the causal analysis conducted in the notebook `03_offline_analysis`, we know the true causes are likely `chamber_temperature`, `chamber_humidity`, or a combination of the two, which were assigned negligible weights above.
# MAGIC
# MAGIC Now let's take a look at `shap`:

# COMMAND ----------

import shap

# Use SHAP to explain a prediction for a sample with quality = 1
explainer = shap.Explainer(model, X_train)
shap_values = explainer(samples)

# Display SHAP explanation
display(shap.plots.waterfall(shap_values[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC The analysis using `shap` yields similar results to `lime`, which is expected since the base classifier (e.g., `XGBoost`) lacks essential information about the causal relationships between the variables. Consequently, it cannot reliably identify the root cause of the variance in the target variable.

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
