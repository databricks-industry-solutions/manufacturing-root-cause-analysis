# Databricks notebook source
import pandas as pd
import numpy as np
from scipy.stats import norm, halfnorm

def generate_data(catalog, schema, n, p_worker=0.75, train=True):
    """
    Generate synthetic manufacturing data and optionally write it to a Delta table.

    This function simulates various features (e.g., raw material, worker skill, 
    machine settings, chamber conditions) and derives multiple quality checks 
    (dimensions, torque checks, visual inspection) to form an overall quality metric. 
    By default, it seeds the random number generator (np.random.seed(1)) for reproducibility.

    Parameters:
        catalog (str): The catalog name where the table will be stored.
        schema (str): The schema name where the data table will be stored.
        n (int): The number of data records to generate.
        p_worker (float, optional): Probability for choosing worker=0 vs. worker=1. Defaults to 0.75.
        train (bool, optional): Whether to write the generated DataFrame to a Delta table. Defaults to True.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the simulated manufacturing data, 
        including features, intermediate metrics, and a final quality indicator.
    """
    
    np.random.seed(1)

    raw_material = np.random.choice([0, 1], size=n, p=[0.75, 0.25])       # Raw material
    material = np.random.choice([0, 1], size=n, p=[0.75, 0.25])           # Additional material
    worker = np.random.choice([0, 1], size=n, p=[p_worker, 1 - p_worker]) # Manual worker
    machine = np.random.choice([0, 1], size=n, p=[0.75, 0.25])            # Machine setting

    chamber_temperature = np.random.normal(loc=20, scale=5, size=n)       # Celsius (°C)
    chamber_humidity = np.random.normal(loc=0.5, scale=0.1, size=n)       # Humidity
    chamber_pressure = np.random.normal(loc=1013.25, scale=15, size=n)    # Atmospheric pressure (hPa)

    X = pd.DataFrame(
        {
            'id': [i + 1 for i in range(n)],
            'raw_material': raw_material,
            'worker': worker,
            'machine': machine,
            'material': material,
            'chamber_temperature': chamber_temperature,
            'chamber_humidity': chamber_humidity,
            'chamber_pressure': chamber_pressure,
        }
    )

    # Measurement of the alignment of the materials and the machine relative to the standard, expressed in millimeters (mm)
    #   worker 1 is less precise than worker 0
    #   machine 1 is less precise than machine 0
    X['position_alignment'] = (
        0.1 + norm.rvs(loc=0, scale=0.005, size=n)  # const + noise
        + halfnorm.rvs(loc=0.1, scale=0.01, size=n) * X['worker']
        + halfnorm.rvs(loc=0.1, scale=0.01, size=n) * X['machine']
    )

    # Measurement of the forces the machine exerts on the materials, expressed in newtons (N)
    #   raw_material 1 is harder than raw_material 0
    #   material 1 is harder than material 0
    #   machine 1 applies stronger forces than machine 0
    X['force_torque'] = (
        500 + norm.rvs(loc=0, scale=25, size=n)    # const + noise
        + halfnorm.rvs(loc=50, scale=5, size=n) * X['raw_material']
        + halfnorm.rvs(loc=50, scale=5, size=n) * X['material']
        + halfnorm.rvs(loc=50, scale=5, size=n) * X['machine']
    )

    # Measurement of the temperature of the materials or the machine, expressed in celsius (°C)
    #   Higher chamber_temperature leads to higher welding temperature
    #   Higher chamber_humidity and chamber_pressure leads to lower welding temperature
    X['temperature'] = (
        1250 + norm.rvs(loc=0, scale=20, size=n)  # const + noise
        + halfnorm.rvs(loc=415, scale=41.5, size=n) * ((X['chamber_temperature'] - 20) / 20)
        - halfnorm.rvs(loc=415, scale=41.5, size=n) * ((X['chamber_humidity'] - 0.5) / 0.5)
        - halfnorm.rvs(loc=415, scale=41.5, size=n) * ((X['chamber_pressure'] - 1013.25) / 1013.25)
    )

    # Dimensional check performed on the processed material, indicated as 0 (pass) or 1 (fail)
    #   Larger position_alignment leads to higher chances of not passing the dimensions check
    #   Lower force_torque leads to larger dimensions
    #   The cutoff is arbitrary
    X['dimensions'] = (X['position_alignment'] - 0.1) / 0.1 - (X['force_torque'] - 500) / 500
    X['dimensions'] = X['dimensions'].apply(lambda x: np.random.choice([0, 1], p=[0.05, 0.95]) if x > 2.0 else 0)

    # Torque-resistance check performed on the processed material, indicated as 0 (pass) or 1 (fail)
    #   Stronger force_torque leads to lower chances of not passing torque_checks
    #   Higher temperature leads to lower chances of not passing torque_checks
    #   The cutoff is arbitrary
    X['torque_checks'] = (X['force_torque'] - 500) / 500 + (X['temperature'] - 1250) / 1250
    X['torque_checks'] = X['torque_checks'].apply(lambda x: np.random.choice([0, 1], p=[0.05, 0.95]) if x < -0.175 else 0)

    # Visual inspection check performed on the processed material, indicated as 0 (pass) or 1 (fail)
    #   Higher welding temperature leads to higher chances of failing the check due to welding spatters
    #   The cutoff is arbitrary
    X['visual_inspection'] = X['temperature'].apply(lambda x: np.random.choice([0, 1], p=[0.05, 0.95]) if x > 1550 else 0)

    # If any of dimensions, torque_checks or visual_inspection fails then the quality_check is negative
    X['quality'] = X.apply(lambda x: 1 if x['dimensions'] + x['torque_checks'] + x['visual_inspection'] > 0 else 0, axis=1)

    # Write the dataframe to a delta table 
    if train:
        (
            spark.createDataFrame(X)
            .write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(f"{catalog}.{schema}.data_manufacturing")
        )
      
    return X

# COMMAND ----------

def setup_unity_catalog(catalog, schema):
    """
    Set up Unity Catalog by creating or verifying the existence of a catalog and schema.
    
    Parameters:
        catalog (str): The catalog name to create/verify
        schema (str): The schema name to create/verify
        
    Raises:
        ValueError: If catalog or schema cannot be created and don't exist
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import NotFound, PermissionDenied
    
    w = WorkspaceClient()
    
    # Create UC Catalog if it does not exist
    try:
        _ = w.catalogs.get(catalog)
        print(f"PASS: UC catalog `{catalog}` exists")
    except NotFound as e:
        print(f"`{catalog}` does not exist, trying to create...")
        try:
            _ = w.catalogs.create(name=catalog)
        except PermissionDenied as e:
            print(f"FAIL: `{catalog}` does not exist, and no permissions to create. Please provide an existing UC Catalog.")
            raise ValueError(f"Unity Catalog `{catalog}` does not exist.")
            
    # Create UC Schema if it does not exist
    try:
        _ = w.schemas.get(full_name=f"{catalog}.{schema}")
        print(f"PASS: UC schema `{catalog}.{schema}` exists")
    except NotFound as e:
        print(f"`{catalog}.{schema}` does not exist, trying to create...")
        try:
            _ = w.schemas.create(name=schema, catalog_name=catalog)
        except PermissionDenied as e:
            print(f"FAIL: `{catalog}.{schema}` does not exist, and no permissions to create. Please provide an existing UC Schema.")
            raise ValueError(f"Unity Catalog Schema `{catalog}.{schema}` does not exist.")
