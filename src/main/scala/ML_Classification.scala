import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler}
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._

val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("C:/Users/lolef/stackoverflow_final_transformed.csv")

val cleanDF = df.filter(
  col("DevType_segment_indexed").isNotNull &&
  col("Age_encoded").isNotNull &&
  col("YearsCoding_encoded").isNotNull &&
  col("JobSatisfaction_encoded").isNotNull &&
  col("Country_indexed").isNotNull &&
  col("Employment_indexed").isNotNull &&
  col("Gender_indexed").isNotNull &&
  col("UndergradMajor_indexed").isNotNull &&
  col("Languages_count").isNotNull &&
  col("EducationTypes_count").isNotNull &&
  col("Salary_num").isNotNull
)

val Array(trainData, testData) = cleanDF.randomSplit(Array(0.7, 0.3), seed = 42)

val totalRows  = cleanDF.count()
val trainCount = trainData.count()
val testCount  = testData.count()

println("=== Dataset Loaded ===")
println(s"Total rows (clean): $totalRows")
println(s"Train size: $trainCount")
println(s"Test size:  $testCount")

val featureCols = Array(
  "Age_encoded",
  "YearsCoding_encoded",
  "JobSatisfaction_encoded",
  "Country_indexed",
  "Employment_indexed",
  "Gender_indexed",
  "UndergradMajor_indexed",
  "Languages_count",
  "EducationTypes_count",
  "Salary_num"
)

val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")

val scaler = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("scaledFeatures")
  .setWithMean(true)
  .setWithStd(true)

println("\n" + "=" * 60)
println("BASELINE: Majority Class")
println("=" * 60)

val majorityClass = trainData
  .groupBy("DevType_segment_indexed")
  .count()
  .orderBy(desc("count"))
  .first()
  .getDouble(0)

val majorityCount    = testData.filter(col("DevType_segment_indexed") === majorityClass).count().toDouble
val baselineAccuracy = majorityCount / testCount.toDouble

println(f"Majority class index: $majorityClass")
println(f"Baseline Accuracy: ${baselineAccuracy * 100}%.2f%%")

println("\n" + "=" * 60)
println("MODEL 1: Logistic Regression")
println("=" * 60)

val lr = new LogisticRegression()
  .setLabelCol("DevType_segment_indexed")
  .setFeaturesCol("scaledFeatures")
  .setMaxIter(100)

val lrPipeline    = new Pipeline().setStages(Array(assembler, scaler, lr))
val lrModel       = lrPipeline.fit(trainData)
val lrPredictions = lrModel.transform(testData)

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("DevType_segment_indexed")
  .setPredictionCol("prediction")

val lrAccuracy = evaluator.setMetricName("accuracy").evaluate(lrPredictions)
val lrF1       = evaluator.setMetricName("f1").evaluate(lrPredictions)

println(f"Logistic Regression Accuracy: ${lrAccuracy * 100}%.2f%%")
println(f"Logistic Regression F1-Score: ${lrF1 * 100}%.2f%%")

println("\n" + "=" * 60)
println("MODEL 2: Random Forest Classifier")
println("=" * 60)

val rf = new RandomForestClassifier()
  .setLabelCol("DevType_segment_indexed")
  .setFeaturesCol("features")
  .setNumTrees(50)
  .setMaxDepth(8)
  .setMaxBins(64)
  .setSeed(42)

val rfPipeline    = new Pipeline().setStages(Array(assembler, rf))
val rfModel       = rfPipeline.fit(trainData)
val rfPredictions = rfModel.transform(testData)

val rfAccuracy = evaluator.setMetricName("accuracy").evaluate(rfPredictions)
val rfF1       = evaluator.setMetricName("f1").evaluate(rfPredictions)

println(f"Random Forest Accuracy: ${rfAccuracy * 100}%.2f%%")
println(f"Random Forest F1-Score: ${rfF1 * 100}%.2f%%")

println("\n" + "=" * 60)
println("CONFUSION MATRIX: Random Forest")
println("=" * 60)

val rfRDD = rfPredictions
  .select("prediction", "DevType_segment_indexed")
  .rdd
  .map(row => (row.getDouble(0), row.getDouble(1)))

val metrics = new MulticlassMetrics(rfRDD)
println("Confusion Matrix:")
println(metrics.confusionMatrix)

println("\nPer-class Precision / Recall / F1:")
metrics.labels.foreach { label =>
  println(f"  Class $label%.0f -> " +
    f"Precision: ${metrics.precision(label)}%.3f | " +
    f"Recall: ${metrics.recall(label)}%.3f | " +
    f"F1: ${metrics.fMeasure(label)}%.3f")
}

println("\n" + "=" * 60)
println("FEATURE IMPORTANCE: Random Forest")
println("=" * 60)

val rfStageModel = rfModel.stages(1)
  .asInstanceOf[org.apache.spark.ml.classification.RandomForestClassificationModel]

val importances = rfStageModel.featureImportances.toArray
val featureImportanceDF = spark.createDataFrame(
  featureCols.zip(importances).map { case (name, imp) => (name, imp) }
).toDF("feature", "importance")
  .orderBy(desc("importance"))

featureImportanceDF.show(truncate = false)

println("\n" + "=" * 60)
println("SAVING MODELS AND OUTPUTS")
println("=" * 60)

lrModel.save("C:/Users/lolef/models/logistic_regression_model")
rfModel.save("C:/Users/lolef/models/random_forest_model")

featureImportanceDF
  .write
  .mode("overwrite")
  .csv("C:/Users/lolef/output/feature_importance")

rfPredictions
  .select("DevType_segment_indexed", "prediction")
  .write
  .mode("overwrite")
  .csv("C:/Users/lolef/output/rf_predictions")

println("Models saved to: C:/Users/lolef/models/")
println("Outputs saved to: C:/Users/lolef/output/")

println("\n" + "=" * 60)
println("SUMMARY COMPARISON")
println("=" * 60)
println(f"Baseline (Majority Class) Accuracy : ${baselineAccuracy * 100}%.2f%%")
println(f"Logistic Regression       Accuracy : ${lrAccuracy * 100}%.2f%%  | F1: ${lrF1 * 100}%.2f%%")
println(f"Random Forest             Accuracy : ${rfAccuracy * 100}%.2f%%  | F1: ${rfF1 * 100}%.2f%%")
