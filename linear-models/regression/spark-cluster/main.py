import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import log, exp
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def run_spark_job():
    # --- Step 1: Correctly initialize SparkSession to connect to the cluster ---
    spark = SparkSession.builder \
        .appName("ScalableLinearRegression") \
        .master("spark://spark-master:7077") \
        .getOrCreate()

    print(f"Spark Session initialized. Spark version: {spark.version}")

    # --- Step 2: Load/Create Data ---
    print("Creating simulated large dataset...")
    n_samples = 1_000_000
    n_features = 50
    feature_names = [f'feature_{i}' for i in range(n_features)]
    data = {name: np.random.rand(n_samples) for name in feature_names}
    data['SalePrice'] = np.exp(12 + data['feature_0'] * 2 + np.random.normal(0, 0.25, n_samples))
    train_df_pandas = pd.DataFrame(data)
    train_df_spark = spark.createDataFrame(train_df_pandas)
    train_df_spark.cache()

    # --- Step 3: Feature Engineering ---
    print("Preparing feature engineering pipeline...")
    train_df_spark = train_df_spark.withColumn("logSalePrice", log(train_df_spark["SalePrice"]))
    feature_cols = [col for col in train_df_spark.columns if col not in ['SalePrice', 'logSalePrice']]
    vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features")
    scaler = StandardScaler(inputCol="unscaled_features", outputCol="features")

    # --- Step 4 & 5: Train and Evaluate a Single Model ---
    (training_data, test_data) = train_df_spark.randomSplit([0.8, 0.2], seed=42)
    
    lr = LinearRegression(featuresCol="features", labelCol="logSalePrice", regParam=0.1, elasticNetParam=0.8)
    pipeline = Pipeline(stages=[vector_assembler, scaler, lr])

    print("Training single pipeline model...")
    pipeline_model = pipeline.fit(training_data)

    print("Evaluating single model...")
    predictions = pipeline_model.transform(test_data)
    predictions = predictions.withColumn("originalPrediction", exp(predictions["prediction"]))
    
    evaluator_rmse = RegressionEvaluator(labelCol="SalePrice", predictionCol="originalPrediction", metricName="rmse")
    rmse = evaluator_rmse.evaluate(predictions)
    print(f"RMSE on test data for single model = ${rmse:,.2f}")


    # --- Step 6: Hyperparameter Tuning with CrossValidator ---
    lr_tuning = LinearRegression(featuresCol="features", labelCol="logSalePrice")
    pipeline_tuning = Pipeline(stages=[vector_assembler, scaler, lr_tuning])
    
    paramGrid = ParamGridBuilder() \
        .addGrid(lr_tuning.regParam, [0.01, 0.1]) \
        .addGrid(lr_tuning.elasticNetParam, [0.5, 0.8]) \
        .build()
        
    evaluator_cv = RegressionEvaluator(labelCol="logSalePrice", predictionCol="prediction", metricName="rmse")
    
    crossval = CrossValidator(estimator=pipeline_tuning, estimatorParamMaps=paramGrid, evaluator=evaluator_cv, numFolds=3)

    print("\nStarting hyperparameter tuning with CrossValidator...")
    cv_model = crossval.fit(training_data)
    
    print("Evaluating best model from tuning...")
    best_predictions = cv_model.transform(test_data)
    best_predictions = best_predictions.withColumn("originalPrediction", exp(best_predictions["prediction"]))
    best_rmse = evaluator_rmse.evaluate(best_predictions)
    print(f"\nRMSE on test data after tuning = ${best_rmse:,.2f}")

    # --- Step 7: Stop the Spark Session ---
    print("Job complete. Stopping Spark session.")
    spark.stop()

# This makes the script runnable
if __name__ == "__main__":
    run_spark_job()