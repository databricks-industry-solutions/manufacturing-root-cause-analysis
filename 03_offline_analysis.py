# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/manufacturing-root-cause-analysis/).

# COMMAND ----------

# MAGIC %md
# MAGIC # Answer causal questions
# MAGIC We have reached the climax of this project! The ultimate goal of this use case was to uncover the true root causes behind a drop in quality in a specific product or a batch of products. In this notebook, we will:
# MAGIC
# MAGIC 1. Conduct root cause analysis on a few specific products flagged as defective for various reasons.
# MAGIC 2. Perform root cause analysis on a batch of products exhibiting a higher defect rate compared to the training dataset.
# MAGIC
# MAGIC Refer to the notebook `01_causal_graph` for the recommended cluster configuration.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies

# COMMAND ----------

# DBTITLE 1,Install graphviz from nicer visualization
# MAGIC %sh apt-get update && apt-get install -y graphviz graphviz-dev

# COMMAND ----------

# MAGIC %md
# MAGIC We install the required packages from the `requirements.txt`.

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
from mlflow import MlflowClient
import pandas as pd
import numpy as np
from dowhy import gcm
from dowhy.utils import bar_plot

mlflow.autolog(disable=True)  # Disabling MLflow autolog as we are just trying to optimise the process and don't need to log everything
gcm.config.disable_progress_bars()  # We turn off the progress bars here to reduce the number of outputs.

# COMMAND ----------

user_name = spark.sql("SELECT current_user()").collect()[0][0]
first_name = user_name.split(".")[0]

# Set up Unity Catalog
catalog = f'causal_solacc_{first_name}'     # Change this to your catalog name
schema = f'rca'                             # Change this to your schema name
model = f"manufacturing_rca"                # Change this to your model name

setup_unity_catalog(catalog, schema)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the causal model
# MAGIC
# MAGIC Let's load the fitted causal model from the previous notebook. We will use the "champion" alias to ensure the correct version is loaded.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_model_name = f"{catalog}.{schema}.{model}"
model = f"models:/{registered_model_name}@champion"

# Load model as a PyFuncModel
loaded_model = mlflow.pyfunc.load_model(model)
loaded_scm = loaded_model.unwrap_python_model().load_scm()
loaded_causal_graph = loaded_model.unwrap_python_model().load_causal_graph()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the training data

# COMMAND ----------

train = spark.read.table(f"{catalog}.{schema}.data_manufacturing")
train = train.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conduct root cause analysis
# MAGIC
# MAGIC As mentioned earlier, we will explore two scenarios for causal investigation:
# MAGIC
# MAGIC 1. Identifying the root causes of specific defective samples.
# MAGIC 2. Determining the root causes of a batch of samples with a high defect rate.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1. What are the key factors causing the quality drop of a particular product?
# MAGIC
# MAGIC Performing this analysis requires new samples of defective products. For this, we will use the method [`gcm.draw_samples`](https://www.pywhy.org/dowhy/v0.9.1/user_guide/gcm_based_inference/draw_samples.html), which enables us to generate new samples using the fitted causal graph.

# COMMAND ----------

np.random.seed(1)
new_batch = gcm.draw_samples(loaded_scm, num_samples=100)
new_batch.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Above we generated 100 samples that consist of multiple defectove products (`quality = 1`). These originate from instances where either `dimension = 1`, `torque_checks = 1`, or `visual_inspection = 1`. To pinpoint the factors contributing to the quality drop, we use DoWhy's [confidence intervals feature](https://www.pywhy.org/dowhy/v0.9.1/user_guide/gcm_based_inference/estimating_confidence_intervals.html). This feature requires specifying the target node of interest (`quality`) and the anomaly sample to analyze.

# COMMAND ----------

# MAGIC %md
# MAGIC In the first example, we will see the root causes of a product that failed the **dimensional verification**. For this, we select one defective sample from the generated batch that did not pass the dimensional verification.

# COMMAND ----------

defect_dimensions = new_batch[new_batch['dimensions'] == 1].iloc[0]
display(defect_dimensions)

# COMMAND ----------

# Compute confidence intervals for anomaly attributions
median_attributions, confidence_intervals = gcm.confidence_intervals(
    gcm.fit_and_compute(
        gcm.attribute_anomalies,  # Function to attribute anomalies
        loaded_scm,  # Structural causal model
        bootstrap_training_data=train,  # Training data for bootstrapping
        target_node='quality',  # Target node for anomaly attribution
        anomaly_samples=pd.DataFrame([defect_dimensions])  # Anomaly samples to analyze
    ),
    num_bootstrap_resamples=10  # Number of bootstrap resamples
)

# Plot the anomaly attribution scores with confidence intervals
bar_plot(median_attributions, confidence_intervals, 'Anomaly attribution score')

# COMMAND ----------

# MAGIC %md
# MAGIC The bar chart above shows the anomaly attribution scores for the nodes associated with a product that failed the quality check due to dimensional verification. Positive values represent nodes that increased the likelihood of the sample being an anomaly, while negative values indicate the opposite. The confidence intervals reflect the uncertainty in the results, stemming from the fitted model parameters and algorithmic approximations. More details about the interpretation of the score can be found in the corresponding [research paper](https://proceedings.mlr.press/v162/budhathoki22a.html).
# MAGIC
# MAGIC Notably, `worker` and `machine` stand out as the primary factors influencing `quality`. This aligns with our synthetic data generation, where `worker = 1` and `machine = 1` were designed to be less precise in positioning and aligning materials and equipment. Interestingly, other factors, such as `position_alignment`, also seem to impact `quality`, as indicated by its positive contribution. However, this factor was not explicitly modeled (latent variable) in our causal graph and was instead included as a source of noise in the data generation process (see the method `generate_data` in the notebook `99_utils` for more information).
# MAGIC
# MAGIC If we observe numerous defective products with a similar root cause structure, we can confidently intervene by refining machine calibration protocols or introducing enhanced worker training programs.

# COMMAND ----------

# MAGIC %md
# MAGIC In the second example, we will see the root causes of a product that failed the **torque resistance check**. For this, we select one defective sample from the generated batch that did not pass the torque checks.

# COMMAND ----------

defect_torque_checks = new_batch[new_batch['torque_checks'] == 1].iloc[0]
display(defect_torque_checks)

# COMMAND ----------

median_attributions, confidence_intervals, = gcm.confidence_intervals(
    gcm.fit_and_compute(
        gcm.attribute_anomalies,
        loaded_scm,
        bootstrap_training_data=train,
        target_node='quality',
        anomaly_samples=pd.DataFrame([defect_torque_checks])
        ),
    num_bootstrap_resamples=10
    )
    
bar_plot(median_attributions, confidence_intervals, 'Anomaly attribution score')

# COMMAND ----------

# MAGIC %md
# MAGIC This time, `chamber_humidity` emerges as the primary factor influencing `quality`. This aligns with our synthetic data generation logic, where higher `chamber_humidity` was designed to lower interface `temperature` due to condensation and evaporation effects. Examining the values for this sample—`chamber_humidity` (0.79) and `temperature` (899)—reveals significant deviations from their expected standard values (0.5 and 1250, respectively; see the `generate_data` function in the `99_utils` notebook for more details). These deviations ultimately led to the torque resistance failure, as the lower interface `temperature` resulted in weak bonds.
# MAGIC
# MAGIC A potential countermeasure for this issue is to install equipment designed to minimize humidity fluctuations within the processing chamber.

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we will see the root causes of a product that failed the **visual inspection**. For this, we select one defective sample from the generated batch that did not pass the visual inspection.

# COMMAND ----------

defect_visual_inspection = new_batch[new_batch['visual_inspection'] == 1].iloc[0]
display(defect_visual_inspection)

# COMMAND ----------

median_attributions, confidence_intervals, = gcm.confidence_intervals(
    gcm.fit_and_compute(
        gcm.attribute_anomalies,
        loaded_scm,
        bootstrap_training_data=train,
        target_node='quality',
        anomaly_samples=pd.DataFrame([defect_visual_inspection])
        ),
    num_bootstrap_resamples=10
    )
    
bar_plot(median_attributions, confidence_intervals, 'Anomaly attribution score')

# COMMAND ----------

# MAGIC %md
# MAGIC Our final analysis reveals that `chamber_humidity` and `chamber_temperature` are the primary factors influencing `quality`. Interestingly, unlike the previous example, this anomaly is attributed to `chamber_humidity` being significantly ***lower*** than the standard. Combined with the elevated `chamber_temperature`, this resulted in a higher interface `temperature`, creating conditions for undesirable outcomes, such as excessive welding spatters, which ultimately caused the visual inspection checks to fail. Additionally, there appears to be an unmodeled factor impacting `temperature`, increasing the likelihood of this sample being classified as an anomaly—something not captured in our causal graph. To prevent similar defects from occurring, we could issue an alert and prompt manual workers to take preventive actions when `chamber_humidity` falls below and `chamber_temperature` rises above certain thresholds simultaneously.
# MAGIC
# MAGIC Causal AI provides transparent identification of attribute combinations contributing to undesired outcomes at the sample level, offering deeper insights compared to traditional correlation-based machine learning approaches.
# MAGIC
# MAGIC ***Note: Given the stochastic nature of sampling, the displayed anomaly distribution might differ slightly from what is discussed here. We encourage users to experiment with the `confidence_intervals` feature using a variety of samples.***

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2. What caused the quality drop in a batch of products?
# MAGIC
# MAGIC In the previous section, we focused on anomaly attribution for a single observation flagged as defective. However, such defects could result from an unfortunate combination of statistical outliers—rare events that occur probabilistically but occasionally. For these rare instances, there may not be many meaningful interventions we can take.
# MAGIC
# MAGIC On the other hand, if we observe a drop in quality across a batch of products rather than just a single sample, it is more likely that a shift in the underlying data generation process has caused the quality deterioration. In this section, we will explore a shift in quality across batches of products. To illustrate this, we will simulate a scenario where a new batch of products exhibits a significantly higher defect rate. We will generate a new batch using the funtion `generate_data` in the notebook `99_utils`:

# COMMAND ----------

test = generate_data(catalog, schema, 100, p_worker=0.25, train=False)
test['quality'].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC The defect rate has indeed increased from 0.075 in our training dataset to 0.17. Here’s what happened behind the scenes with the new batch: `worker = 0`, who previously handled 75% of the products, is on leave, and `worker = 1` has taken over her role. In this new batch, `worker = 1` processes 75% of the products as a manual operator, while `worker = 0` handles only 25%. We can simulate this distributional drift using the parameter `p_worker` in the function `generate_data`.
# MAGIC
# MAGIC Let’s now see if DoWhy can accurately identify the root cause of this change in the batch defect rate. We will apply the [distribution change method](https://proceedings.mlr.press/v130/budhathoki21a.html) to identify the part in the system that has changed:

# COMMAND ----------

median_attributions, confidence_intervals = gcm.confidence_intervals(
    lambda: gcm.distribution_change(loaded_scm,
                                    train,
                                    test,
                                    target_node='quality',
                                    # Here, we are intersted in explaining the differences in the mean.
                                    difference_estimation_func=lambda x, y: np.mean(y) - np.mean(x)) 
)

bar_plot(median_attributions, confidence_intervals, 'Change attribution in defect rate')

# COMMAND ----------

# MAGIC %md
# MAGIC The distribution change method accurately identifies the root cause of the increased defect rate, clearly indicating that the shift in `worker` has significantly contributed to this change. Since this is likely not a statistical fluke (given the aggregation over 100 samples), we can now confidently propose an intervention plan to enhance operational efficiency.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Wrap up
# MAGIC
# MAGIC In this notebook, we addressed the ultimate goal of our use case: uncovering the true root causes behind quality drops. We conducted root cause analysis on individual products flagged as defective for various reasons. We aslo analyzed a batch of products with a higher defect rate compared to the training dataset, identifying shifts in the underlying data generation processes. By pinpointing the key factors contributing to defects, we pave the way for targeted interventions that can enhance overall product quality and operational efficiency.
# MAGIC
# MAGIC In the next notebook, we will deploy the fitted graph to Databricks Model Serving and explore how to enable real-time causal analysis.

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
