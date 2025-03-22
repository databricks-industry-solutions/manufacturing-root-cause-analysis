# Databricks notebook source
# MAGIC %md
# MAGIC This solution accelerator notebook is available at [Databricks Industry Solutions](https://github.com/databricks-industry-solutions/).

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://github.com/databricks-industry-solutions/.github/raw/main/profile/solacc_logo_wide.png' width="1000" ></img>
# MAGIC
# MAGIC # Manufacturing Root Cause Analysis Using Causal AI
# MAGIC
# MAGIC Root cause analysis in manufacturing is essential for uncovering the underlying issues that result in defects, inefficiencies, and failures. By identifying the true sources of problems, manufacturers can implement targeted solutions to prevent recurrence, improving product quality, reducing waste, and enhancing operational efficiency. Additionally, effective root cause analysis ensures compliance with industry standards, minimizes safety risks, and bolsters a manufacturer’s competitive edge in the market.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Challenges with traditional machine learning approaches
# MAGIC
# MAGIC Many manufacturers rely on traditional machine learning techniques based on correlations to address this problem. However, these techniques have significant limitations in root cause analysis due to their inability to capture causality. They often fail to distinguish true root causes from mere symptoms, oversimplifying complex manufacturing processes into tabular data while neglecting the manufacturing process flows. By prioritizing predictive power over causal understanding, these algorithms risk misidentifying root causes. Consequently, they can lead to misleading conclusions. See the section `Appendix B` in the notebook `05_appendix` for more information.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enhancing root cause analysis with causal AI
# MAGIC
# MAGIC Causal AI enhances root cause analysis in manufacturing by modeling cause-and-effect relationships within complex production processes, moving beyond mere correlations. It utilizes domain knowledge, often represented as knowledge graphs, to capture the causal relationships among critical variables in manufacturing. This approach distinguishes actual root causes from symptoms, enabling more precise identification of issues and their origins. By integrating observational data from production lines with causal insights, causal AI offers actionable recommendations for defect prevention and process optimization.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Case study
# MAGIC
# MAGIC In this series of notebooks, we demonstrate how causal AI can be applied to perform root cause analysis in a manufacturing process. We create a fictitious scenario where we are responsible for reducing costs and optimizing the efficiency of a production line. Through this setup, we explore how various factors influence the quality of final products.
# MAGIC
# MAGIC <img src="images/manufacturing-process-A.png" alt="Simplified Flow Between Production Line Processes" width="1000">
# MAGIC
# MAGIC Above is a schematic representation of our production line. While real-world production lines are often far more complex and involve significantly larger number of variables, this simplified case serves as a practical starting point for building intuition about the technique.

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
