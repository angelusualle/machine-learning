from pyspark.sql import SparkSession
from pyspark.ml.feature import FeatureHasher
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

NUM_FEATURES = 10194304
MODEL_ID = "trained_1e-4_default_max_iter_10m_feats"
spark = SparkSession.builder.appName("TrainAndHyperParameterTune") \
    .master(f"spark://spark-master:7077") \
    .config("spark.kryoserializer.buffer.max", "256m") \
    .getOrCreate()
df = spark.read.parquet(".data/preprocessed/train_2k_partitions")


columns_to_exclude = {"click", "id"}
categorical_cols = [col for col in df.columns if col not in columns_to_exclude]

hasher = FeatureHasher(
    inputCols=categorical_cols,
    outputCol="features",
    numFeatures=NUM_FEATURES
)

lr = LogisticRegression(
    featuresCol="features", 
    labelCol="click",
    elasticNetParam=1.0,  
    regParam=0.0001

)

pipeline = Pipeline(stages=[hasher, lr])

print(f"Training Logistic Regression with {str(NUM_FEATURES)} Hash Buckets...")
model = pipeline.fit(df)

model.save(f".data/model/{MODEL_ID}")
print("Done training")

lr_model = model.stages[-1]
training_summary = lr_model.summary

print(f"Training ROC-AUC: {training_summary.areaUnderROC}")

print(f"Final Training Log Loss: {training_summary.objectiveHistory[-1]}")


test_df = spark.read.parquet(".data/preprocessed/test") 


print("Running through test data")
test_predictions = model.transform(test_df)


evaluator_roc = BinaryClassificationEvaluator(
    labelCol="click", 
    rawPredictionCol="rawPrediction", 
    metricName="areaUnderROC"
)

evaluator_pr = BinaryClassificationEvaluator(
    labelCol="click", 
    rawPredictionCol="rawPrediction", 
    metricName="areaUnderPR"
)

evaluator_logloss = MulticlassClassificationEvaluator(
    labelCol="click", 
    probabilityCol="probability", 
    metricName="logLoss"
)

test_roc_auc = evaluator_roc.evaluate(test_predictions)
print(f"Test ROC-AUC:  {test_roc_auc}")

test_pr_auc = evaluator_pr.evaluate(test_predictions)
print(f"Test PR-AUC:   {test_pr_auc}")

test_logloss = evaluator_logloss.evaluate(test_predictions)
print(f"Test Log Loss: {test_logloss}")
spark.stop()
