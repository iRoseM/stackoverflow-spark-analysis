import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions._

// ============================================================
// SPARK CONFIGURATION
// ============================================================

spark.conf.set("spark.sql.shuffle.partitions", "100")
spark.conf.set("spark.default.parallelism", "100")
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

// ============================================================
// LOAD & CLEAN DATASET
// ============================================================

val df = spark.read
  .option("header", "true")
  .option("inferSchema", "true")
  .csv("C:/Users/lolef/Downloads/stackoverflow_final_transformed.csv")

val cleanDF = df.filter(
  col("DevType_segment").isNotNull &&
  col("Age_encoded").isNotNull &&
  col("YearsCoding_encoded").isNotNull &&
  col("JobSatisfaction_encoded").isNotNull &&
  col("Employment").isNotNull &&
  col("Gender").isNotNull &&
  col("UndergradMajor").isNotNull &&
  col("Languages_count").isNotNull &&
  col("EducationTypes_count").isNotNull &&
  col("Salary_num").isNotNull
)

println("=== Dataset Loaded ===")
println(s"Total rows (clean): ${cleanDF.count()}")
println("Class distribution:")
cleanDF.groupBy("DevType_segment").count().orderBy(desc("count")).show(truncate = false)

// ============================================================
// TRAIN / TEST SPLIT
// ============================================================

val Array(trainRaw, testData) = cleanDF.randomSplit(Array(0.7, 0.3), seed = 42)

testData.persist()
testData.count()

println(s"Train size: ${trainRaw.count()}")
println(s"Test size:  ${testData.count()}")

// ============================================================
// CLASS WEIGHTS
// ============================================================

val totalRows  = trainRaw.count().toDouble
val numClasses = trainRaw.select("DevType_segment").distinct().count().toDouble

val classWeightDF = trainRaw
  .groupBy("DevType_segment")
  .count()
  .withColumn("classWeight", lit(totalRows) / (lit(numClasses) * col("count")))
  .select("DevType_segment", "classWeight")

val trainWithWeights = trainRaw
  .join(classWeightDF, Seq("DevType_segment"))
  .repartition(100)
  .persist()

trainWithWeights.count()

println("Class Weights:")
classWeightDF.orderBy(desc("classWeight")).show(truncate = false)

// ============================================================
// BASELINE
// ============================================================

val majorityClass    = classWeightDF.orderBy("classWeight").first().getString(0)
val baselineAccuracy = testData
  .filter(col("DevType_segment") === majorityClass).count().toDouble / testData.count().toDouble

println("\n" + "=" * 60)
println("BASELINE: Majority Class Classifier")
println("=" * 60)
println(f"Majority class   : $majorityClass")
println(f"Baseline Accuracy: ${baselineAccuracy * 100}%.2f%%")

// ============================================================
// FEATURE ENGINEERING
// ============================================================

val stringCols = Array("Employment", "Gender", "UndergradMajor")

val indexers = stringCols.map(c =>
  new StringIndexer()
    .setInputCol(c)
    .setOutputCol(c + "_idx")
    .setHandleInvalid("keep")
)

val encoders = stringCols.map(c =>
  new OneHotEncoder()
    .setInputCol(c + "_idx")
    .setOutputCol(c + "_vec")
)

val labelIndexer = new StringIndexer()
  .setInputCol("DevType_segment")
  .setOutputCol("label")
  .setHandleInvalid("keep")

val featureCols = Array(
  "Age_encoded",
  "YearsCoding_encoded",
  "JobSatisfaction_encoded",
  "Employment_vec",
  "Gender_vec",
  "UndergradMajor_vec",
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

// ============================================================
// MODEL 1: LOGISTIC REGRESSION
// ============================================================

val lr = new LogisticRegression()
  .setLabelCol("label")
  .setFeaturesCol("scaledFeatures")
  .setMaxIter(100)
  .setRegParam(0.01)
  .setElasticNetParam(0.5)
  .setWeightCol("classWeight")

val lrPipeline = new Pipeline()
  .setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, scaler, lr))

println("\nTraining Logistic Regression...")
val lrModel       = lrPipeline.fit(trainWithWeights)
val lrPredictions = lrModel.transform(testData).persist()

val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")

val lrAccuracy = evaluator.setMetricName("accuracy").evaluate(lrPredictions)
val lrF1       = evaluator.setMetricName("f1").evaluate(lrPredictions)

val lrRDD = lrPredictions
  .select("prediction", "label")
  .rdd.map(r => (r.getDouble(0), r.getDouble(1)))
val lrMetrics = new MulticlassMetrics(lrRDD)

val lrMacroF1 = lrMetrics.labels.map(lrMetrics.fMeasure).sum / lrMetrics.labels.length

println("\n" + "=" * 60)
println("MODEL 1: Logistic Regression")
println("=" * 60)
println(f"Accuracy           : ${lrAccuracy * 100}%.2f%%")
println(f"Weighted F1        : ${lrF1 * 100}%.2f%%")
println(f"Macro F1           : ${lrMacroF1 * 100}%.2f%%")
println(f"Weighted Precision : ${lrMetrics.weightedPrecision * 100}%.2f%%")
println(f"Weighted Recall    : ${lrMetrics.weightedRecall * 100}%.2f%%")
println("\nPer-Class F1 (LR):")
lrMetrics.labels.foreach { l =>
  val p = lrMetrics.precision(l)
  val r = lrMetrics.recall(l)
  val f = lrMetrics.fMeasure(l)
  println(f"  Label $l%.0f  ->  Precision: $p%.3f  Recall: $r%.3f  F1: $f%.3f")
}

// ============================================================
// LR COEFFICIENTS
// ============================================================

val lrStageModel = lrModel.stages
  .collectFirst { case m: LogisticRegressionModel => m }
  .getOrElse(throw new RuntimeException("LR model not found"))

val lrSampleTransformed = lrModel.transform(testData.limit(1))
val lrFeatureAttrs = org.apache.spark.ml.attribute.AttributeGroup
  .fromStructField(lrSampleTransformed.schema("scaledFeatures"))

val lrFeatureNames: Array[String] = lrFeatureAttrs.attributes match {
  case Some(attrs) => attrs.map(_.name.getOrElse("unknown"))
  case None        => (0 until lrStageModel.coefficientMatrix.numCols).map(i => s"feature_$i").toArray
}

val coeffMatrix  = lrStageModel.coefficientMatrix
val numLRClasses = coeffMatrix.numRows

println("\nLR Coefficients (top 10 per class):")
(0 until numLRClasses).foreach { classIdx =>
  val coeffs = (0 until coeffMatrix.numCols).map(j => lrFeatureNames(j) -> coeffMatrix(classIdx, j))
  val top10  = coeffs.sortBy(-_._2.abs).take(10)
  println(s"\n  Class $classIdx:")
  top10.foreach { case (name, coeff) =>
    println(f"    $name%-60s $coeff%+.4f")
  }
}

val coeffRows = (0 until numLRClasses).flatMap { classIdx =>
  (0 until coeffMatrix.numCols).map { j =>
    (classIdx, lrFeatureNames(j), coeffMatrix(classIdx, j))
  }
}
val coeffDF = spark.createDataFrame(coeffRows).toDF("class", "feature", "coefficient")
coeffDF.write.mode("overwrite").csv("C:/Users/lolef/output/lr_coefficients")
println("\nLR Coefficients saved to: C:/Users/lolef/output/lr_coefficients")

// ============================================================
// MODEL 2: RANDOM FOREST
// ============================================================

val rf = new RandomForestClassifier()
  .setLabelCol("label")
  .setFeaturesCol("features")
  .setNumTrees(50)
  .setMaxDepth(8)
  .setMaxBins(32)
  .setMinInstancesPerNode(5)
  .setSubsamplingRate(0.8)
  .setFeatureSubsetStrategy("sqrt")
  .setWeightCol("classWeight")
  .setSeed(42)

val rfPipeline = new Pipeline()
  .setStages(indexers ++ encoders ++ Array(labelIndexer, assembler, rf))

println("\nTraining Random Forest...")
val rfModel       = rfPipeline.fit(trainWithWeights)
val rfPredictions = rfModel.transform(testData).persist()

val rfAccuracy = evaluator.setMetricName("accuracy").evaluate(rfPredictions)
val rfF1       = evaluator.setMetricName("f1").evaluate(rfPredictions)

val rfRDD = rfPredictions
  .select("prediction", "label")
  .rdd.map(r => (r.getDouble(0), r.getDouble(1)))
val rfMetrics = new MulticlassMetrics(rfRDD)

val rfMacroF1 = rfMetrics.labels.map(rfMetrics.fMeasure).sum / rfMetrics.labels.length

println("\n" + "=" * 60)
println("MODEL 2: Random Forest Classifier")
println("=" * 60)
println(f"Accuracy           : ${rfAccuracy * 100}%.2f%%")
println(f"Weighted F1        : ${rfF1 * 100}%.2f%%")
println(f"Macro F1           : ${rfMacroF1 * 100}%.2f%%")
println(f"Weighted Precision : ${rfMetrics.weightedPrecision * 100}%.2f%%")
println(f"Weighted Recall    : ${rfMetrics.weightedRecall * 100}%.2f%%")

println("\nPer-Class F1 (RF):")
rfMetrics.labels.foreach { l =>
  val p = rfMetrics.precision(l)
  val r = rfMetrics.recall(l)
  val f = rfMetrics.fMeasure(l)
  println(f"  Label $l%.0f  ->  Precision: $p%.3f  Recall: $r%.3f  F1: $f%.3f")
}

println("\nConfusion Matrix (RF):")
println(rfMetrics.confusionMatrix)

// ============================================================
// FEATURE IMPORTANCE
// ============================================================

val rfStageModel = rfModel.stages
  .collectFirst { case m: RandomForestClassificationModel => m }
  .getOrElse(throw new RuntimeException("RandomForestClassificationModel not found"))

val importances = rfStageModel.featureImportances.toArray

val sampleTransformed = rfModel.transform(testData.limit(1))
val featureAttrs = org.apache.spark.ml.attribute.AttributeGroup
  .fromStructField(sampleTransformed.schema("features"))

val realFeatureNames: Array[String] = featureAttrs.attributes match {
  case Some(attrs) => attrs.map(_.name.getOrElse("unknown"))
  case None        => (0 until importances.length).map(i => s"feature_$i").toArray
}

val featureImportanceDF = spark
  .createDataFrame(realFeatureNames.zip(importances))
  .toDF("feature", "importance")
  .orderBy(desc("importance"))

println("\nFeature Importances (RF):")
featureImportanceDF.show(truncate = false)

// ============================================================
// SAVE MODELS & OUTPUTS
// ============================================================

lrModel.write.overwrite().save("C:/Users/lolef/models/logistic_regression_model")
rfModel.write.overwrite().save("C:/Users/lolef/models/random_forest_model")

featureImportanceDF.write.mode("overwrite").csv("C:/Users/lolef/output/feature_importance")
rfPredictions.select("label", "prediction").write.mode("overwrite").csv("C:/Users/lolef/output/rf_predictions")
lrPredictions.select("label", "prediction").write.mode("overwrite").csv("C:/Users/lolef/output/lr_predictions")

println("\nModels saved to : C:/Users/lolef/models/")
println("Outputs saved to: C:/Users/lolef/output/")

// ============================================================
// UNPERSIST
// ============================================================

trainWithWeights.unpersist()
testData.unpersist()
lrPredictions.unpersist()
rfPredictions.unpersist()

// ============================================================
// FINAL SUMMARY
// ============================================================

println("\n" + "=" * 60)
println("SUMMARY COMPARISON")
println("=" * 60)
println(f"Baseline (Majority Class) Accuracy : ${baselineAccuracy * 100}%.2f%%")
println(f"Logistic Regression  Accuracy: ${lrAccuracy * 100}%.2f%%  | Weighted F1: ${lrF1 * 100}%.2f%%  | Macro F1: ${lrMacroF1 * 100}%.2f%%")
println(f"Random Forest        Accuracy: ${rfAccuracy * 100}%.2f%%  | Weighted F1: ${rfF1 * 100}%.2f%%  | Macro F1: ${rfMacroF1 * 100}%.2f%%")
println("\nNote: Macro F1 is the most honest metric for imbalanced multi-class classification.")
