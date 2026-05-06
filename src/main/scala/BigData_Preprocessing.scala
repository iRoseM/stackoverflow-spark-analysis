import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.ml.feature.StringIndexer

object BigDataPreprocessing {

  // ============================================================
  // HELPER FUNCTIONS
  // ============================================================

  /**
   * Check if a column value is considered "missing" (comprehensive check)
   */
  def isMissing(colName: String) = {
    val missingIndicators = Seq(
      "", "na", "n/a", "nan", "null", "NA", "N/A", "NaN",
      "-", "--", "none", "None", "unknown", "Unknown",
      "rather not say", " ", "0"
    ).map(_.toLowerCase)

    col(colName).isNull ||
      (trim(col(colName)) === "") ||
      lower(trim(col(colName))).isin(missingIndicators: _*)
  }

  def main(args: Array[String]): Unit = {

    // ============================================================
    // 1) CREATE SPARK SESSION
    // ============================================================
    val spark = SparkSession.builder()
      .appName("StackOverflow Analysis")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    println(s"Spark version: ${spark.version}")

    import spark.implicits._

    // ============================================================
    // 2) READ DATA
    // ============================================================
    val df0 = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("survey_results_public.csv")

    df0.show(5)
    println(s"df0 rows=${df0.count()}, cols=${df0.columns.length}")

    // ============================================================
    // 3) SELECT REQUIRED COLUMNS
    // ============================================================
    val cols = Seq(
      "Respondent", "Country", "DevType", "Employment", "YearsCoding",
      "Age", "Gender", "JobSatisfaction", "UndergradMajor",
      "EducationTypes", "LanguageWorkedWith", "ConvertedSalary"
    )

    val existingCols = cols.filter(df0.columns.contains)
    val df1 = df0.select(existingCols.map(col): _*)

    println(s"df1 rows=${df1.count()}, cols=${df1.columns.length}")
    println(s"Available columns: ${df1.columns.mkString(", ")}")

    // ============================================================
    // 4) DATA QUALITY CHECK
    // ============================================================

    // -- Check Duplicates --
    if (df1.columns.contains("Respondent")) {
      val dupRespondent = df1.groupBy("Respondent").count()
        .filter("count > 1").count()
      println(s"Duplicate Respondent IDs: $dupRespondent")
      if (dupRespondent > 0) {
        println("Warning: Found duplicate respondents!")
        df1.groupBy("Respondent").count().filter("count > 1").show(5)
      }
    } else {
      println("Warning: No 'Respondent' column found")
    }

    val dupFull = df1.count() - df1.distinct().count()
    println(s"Complete duplicate rows: $dupFull")

    // -- Check Missing Values --
    println("\nMISSING VALUES BY COLUMN:")
    println("-" * 80)
    println(f"${"Column Name"}%-30s ${"Missing Count"}%15s ${"Percentage"}%15s")
    println("-" * 80)

    val totalRows = df1.count()
    df1.columns.foreach { colName =>
      val missingCount = df1.filter(isMissing(colName)).count()
      val missingPct = (missingCount.toDouble / totalRows) * 100
      println(f"$colName%-30s $missingCount%15,d $missingPct%14.2f%%")
    }
    println("-" * 80)

    // ============================================================
    // 5) HANDLE MISSING VALUES
    // ============================================================

    val originalCount = df0.count()

    // Drop rows with missing critical columns (Country / DevType / Employment)
    val df2 = df1.filter(
      !isMissing("Country") &&
      !isMissing("DevType") &&
      !isMissing("Employment")
    )
    println(s"df2 rows after dropping key-missing = ${df2.count()}")

    // -- Handle DevType --
    val dfStep1 = df2.filter(!isMissing("DevType"))
    val step1Count = dfStep1.count()
    println(s"Rows deleted (missing DevType): ${originalCount - step1Count}")
    println(s"Rows remaining: $step1Count (${step1Count.toDouble / originalCount * 100:.1f}%)")

    // -- Handle Salary --
    val salaryMean = dfStep1
      .filter(!isMissing("ConvertedSalary"))
      .select(col("ConvertedSalary").cast("double").alias("salary"))
      .filter(col("salary") > 0)
      .agg(mean("salary"))
      .collect()(0)
      .getDouble(0)

    println(f"Valid salary mean: $$$salaryMean%,.2f")

    val dfStep2 = dfStep1.withColumn(
      "ConvertedSalary",
      when(
        isMissing("ConvertedSalary") || col("ConvertedSalary").cast("double") <= 0,
        salaryMean
      ).otherwise(col("ConvertedSalary").cast("double"))
    )

    val zerosLeft = dfStep2.filter(col("ConvertedSalary") === 0).count()
    println(s"Zeros after filling: $zerosLeft")

    // -- Handle EducationTypes, UndergradMajor, Gender --
    val dfStep3 = dfStep2
      .withColumn("EducationTypes",
        when(isMissing("EducationTypes"), "Unknown").otherwise(col("EducationTypes")))
      .withColumn("UndergradMajor",
        when(isMissing("UndergradMajor"), "Unknown").otherwise(col("UndergradMajor")))
      .withColumn("Gender",
        when(isMissing("Gender"), "Unknown").otherwise(col("Gender")))

    // -- Handle YearsCoding --
    val dfStep4 = dfStep3.withColumn(
      "YearsCoding",
      when(isMissing("YearsCoding"), "0-2 years").otherwise(col("YearsCoding"))
    )

    // -- Handle JobSatisfaction (fill with mode) --
    val modeJobSatisfaction = dfStep4
      .filter(!isMissing("JobSatisfaction"))
      .groupBy("JobSatisfaction").count()
      .orderBy(desc("count"))
      .first().getString(0)

    println(s"Most common JobSatisfaction: '$modeJobSatisfaction'")

    val dfStep5 = dfStep4.withColumn(
      "JobSatisfaction",
      when(isMissing("JobSatisfaction"), modeJobSatisfaction)
        .otherwise(col("JobSatisfaction"))
    )

    // -- Handle Age (fill with mode) --
    val modeAge = dfStep5
      .filter(!isMissing("Age"))
      .groupBy("Age").count()
      .orderBy(desc("count"))
      .first().getString(0)

    println(s"Most common Age: '$modeAge'")

    val dfStep6 = dfStep5.withColumn(
      "Age",
      when(isMissing("Age"), modeAge).otherwise(col("Age"))
    )

    // -- Handle Employment (fill with mode) --
    val modeEmployment = dfStep6
      .filter(!isMissing("Employment"))
      .groupBy("Employment").count()
      .orderBy(desc("count"))
      .first().getString(0)

    println(s"Most common Employment: '$modeEmployment'")

    val dfStep7 = dfStep6.withColumn(
      "Employment",
      when(isMissing("Employment"), modeEmployment).otherwise(col("Employment"))
    )

    // -- Handle Country (fill with mode) --
    val modeCountry = dfStep7
      .filter(!isMissing("Country"))
      .groupBy("Country").count()
      .orderBy(desc("count"))
      .first().getString(0)

    println(s"Most common Country: '$modeCountry'")

    val dfStep8 = dfStep7.withColumn(
      "Country",
      when(isMissing("Country"), modeCountry).otherwise(col("Country"))
    )

    // -- Handle LanguageWorkedWith (infer from group, delete remaining Unknown) --
    val validLangs = dfStep8
      .filter(!isMissing("LanguageWorkedWith"))
      .select("Country", "DevType", "Employment", "YearsCoding", "Age", "LanguageWorkedWith")

    val langCounts = validLangs
      .groupBy("Country", "DevType", "Employment", "YearsCoding", "Age", "LanguageWorkedWith")
      .count()

    val windowSpecRank = Window
      .partitionBy("Country", "DevType", "Employment", "YearsCoding", "Age")
      .orderBy(desc("count"))

    val mostCommonLangs = langCounts
      .withColumn("rank", row_number().over(windowSpecRank))
      .filter(col("rank") === 1)
      .drop("rank", "count")
      .withColumnRenamed("LanguageWorkedWith", "inferred_language")

    var dfStep9 = dfStep8.join(
      mostCommonLangs,
      Seq("Country", "DevType", "Employment", "YearsCoding", "Age"),
      "left"
    ).withColumn(
      "LanguageWorkedWith",
      when(
        isMissing("LanguageWorkedWith"),
        coalesce(col("inferred_language"), lit("Unknown"))
      ).otherwise(col("LanguageWorkedWith"))
    ).drop("inferred_language")

    println(s"Rows with 'Unknown' language: ${dfStep9.filter(col("LanguageWorkedWith") === "Unknown").count()}")

    val dfStep10 = dfStep9.filter(col("LanguageWorkedWith") =!= "Unknown")
    println(s"Rows after deletion: ${dfStep10.count()}")

    dfStep10.groupBy("LanguageWorkedWith").count()
      .orderBy(desc("count")).show(10)

    dfStep9 = dfStep10

    // ============================================================
    // 6) FINAL VALIDATION
    // ============================================================
    println("\n" + "=" * 60)
    println("FINAL VALIDATION")
    println("=" * 60)

    val finalCount = dfStep9.count()
    println(s"Final rows: $finalCount")
    println(f"Kept ${finalCount.toDouble / originalCount * 100:.1f}%% of original data")

    println("\nRemaining missing values check:")
    val checkCols = Seq(
      "Country", "DevType", "Employment", "YearsCoding", "Age", "Gender",
      "JobSatisfaction", "UndergradMajor", "EducationTypes",
      "LanguageWorkedWith", "ConvertedSalary"
    )

    checkCols.foreach { colName =>
      val remaining = if (colName == "ConvertedSalary")
        dfStep9.filter(col(colName).isNull).count()
      else
        dfStep9.filter(isMissing(colName)).count()

      val status = if (remaining == 0) "CLEAN" else s"⚠ $remaining missing"
      println(f"$colName%-20s : $status")
    }

    // ============================================================
    // 7) OUTLIER DETECTION AND REMOVAL (Z-score on Salary)
    // ============================================================
    println("\n" + "=" * 60)
    println("OUTLIER REMOVAL FOR SALARY")
    println("=" * 60)

    val salaryStats = dfStep9.select(
      mean("ConvertedSalary").alias("mean"),
      stddev("ConvertedSalary").alias("stddev")
    ).collect()(0)

    val meanSal  = salaryStats.getDouble(0)
    val stdSal   = salaryStats.getDouble(1)

    println(f"Mean salary: $$$meanSal%,.2f")
    println(f"Std deviation: $$$stdSal%,.2f")
    println(s"Total rows: $finalCount")

    val dfWithZscore = dfStep9.withColumn(
      "zscore",
      abs((col("ConvertedSalary") - meanSal) / stdSal)
    )

    val outliersCount = dfWithZscore.filter(col("zscore") > 3).count()
    println(s"\nOutliers detected (|z| > 3): $outliersCount")

    val dfStep10Final = dfWithZscore.filter(col("zscore") <= 3).drop("zscore")
    println(s"After outlier removal: ${dfStep10Final.count()} rows")
    println(s"Rows removed: ${dfStep9.count() - dfStep10Final.count()}")

    val newStats = dfStep10Final.select(
      mean("ConvertedSalary").alias("mean"),
      stddev("ConvertedSalary").alias("stddev"),
      min("ConvertedSalary").alias("min"),
      max("ConvertedSalary").alias("max")
    ).collect()(0)

    println(f"\nNew salary statistics:")
    println(f"  Mean:    $$${ newStats.getDouble(0) }%,.2f")
    println(f"  Std dev: $$${ newStats.getDouble(1) }%,.2f")
    println(f"  Min:     $$${ newStats.getDouble(2) }%,.2f")
    println(f"  Max:     $$${ newStats.getDouble(3) }%,.2f")

    println(s"\nFinal dataset:")
    println(s"  Original rows : $originalCount")
    println(s"  Final rows    : ${dfStep10Final.count()}")
    println(f"  Retention rate: ${dfStep10Final.count().toDouble / originalCount * 100:.1f}%%")

    // Save cleaned data
    println("\n" + "=" * 60)
    println("SAVING FINAL CLEANED DATA")
    println("=" * 60)

    dfStep10Final.write
      .option("header", "true")
      .mode("overwrite")
      .csv("stackoverflow_survey_afterCleaning")

    println("Cleaned data saved to: stackoverflow_survey_afterCleaning/")

    // ============================================================
    // 8) RELOAD CLEANED DATA
    // ============================================================
    val dfCleaned = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("stackoverflow_survey_afterCleaning")

    dfCleaned.show(5)
    dfCleaned.printSchema()
    println(s"Rows: ${dfCleaned.count()}, Cols: ${dfCleaned.columns.length}")

    // ============================================================
    // 9) REDUCTION — Keep only modeling-relevant columns
    // ============================================================
    val reducedCols = Seq(
      "Country", "DevType", "Employment", "YearsCoding", "Age", "Gender",
      "JobSatisfaction", "UndergradMajor", "EducationTypes",
      "LanguageWorkedWith", "ConvertedSalary"
    )

    val reducedExisting = reducedCols.filter(dfCleaned.columns.contains)
    val dfReduced = dfCleaned.select(reducedExisting.map(col): _*)

    println(s"df_reduced rows=${dfReduced.count()}, cols=${dfReduced.columns.length}")
    dfReduced.show(5, truncate = false)

    // ============================================================
    // 10) TRANSFORMATION — Feature Engineering + Encoding
    // ============================================================

    // DevType Segmentation (5 classes)
    val dfSeg = dfReduced.withColumn(
      "DevType_segment",
      when(
        lower(col("DevType")).contains("student") ||
        lower(col("DevType")).contains("educator") ||
        lower(col("DevType")).contains("academic researcher"),
        "Student/Academic"
      ).when(
        lower(col("DevType")).contains("front-end developer") ||
        lower(col("DevType")).contains("back-end developer") ||
        lower(col("DevType")).contains("full-stack developer") ||
        lower(col("DevType")).contains("web developer"),
        "Web"
      ).when(
        lower(col("DevType")).contains("mobile developer") ||
        lower(col("DevType")).contains("android") ||
        lower(col("DevType")).contains("ios"),
        "Mobile"
      ).when(
        lower(col("DevType")).contains("data scientist") ||
        lower(col("DevType")).contains("machine learning") ||
        lower(col("DevType")).contains("data or business analyst"),
        "Data"
      ).otherwise("Other")
    )

    dfSeg.groupBy("DevType_segment").count().orderBy(desc("count")).show(truncate = false)

    // Feature Engineering (counts + log salary)
    val dfFe = dfSeg
      .withColumn(
        "Languages_count",
        when(
          col("LanguageWorkedWith").isNull || trim(col("LanguageWorkedWith")) === "",
          lit(0)
        ).otherwise(size(split(col("LanguageWorkedWith"), ";")))
      )
      .withColumn(
        "EducationTypes_count",
        when(
          col("EducationTypes").isNull ||
          trim(col("EducationTypes")) === "" ||
          lower(col("EducationTypes")) === "unknown",
          lit(0)
        ).otherwise(size(split(col("EducationTypes"), ";")))
      )
      .withColumn("Salary_num", col("ConvertedSalary").cast("double"))
      .withColumn(
        "Salary_log",
        when(
          col("ConvertedSalary").cast("double").isNull ||
          col("ConvertedSalary").cast("double") <= 0,
          lit(0.0)
        ).otherwise(log(col("ConvertedSalary").cast("double")))
      )

    // Encoding with StringIndexer (frequency-based)
    val indexerConfigs = Seq(
      ("DevType_segment",  "DevType_segment_indexed"),
      ("Country",          "Country_indexed"),
      ("Employment",       "Employment_indexed"),
      ("Gender",           "Gender_indexed"),
      ("UndergradMajor",   "UndergradMajor_indexed"),
      ("Age",              "Age_encoded"),
      ("YearsCoding",      "YearsCoding_encoded"),
      ("JobSatisfaction",  "JobSatisfaction_encoded")
    )

    var dfEnc = dfFe
    for ((inputCol, outputCol) <- indexerConfigs) {
      val indexer = new StringIndexer()
        .setInputCol(inputCol)
        .setOutputCol(outputCol)
        .setHandleInvalid("keep")
      dfEnc = indexer.fit(dfEnc).transform(dfEnc)
    }

    // Final model-ready dataset
    val dfFinalModel = dfEnc.select(
      // Raw columns
      col("Country"), col("DevType"), col("Employment"), col("YearsCoding"),
      col("Age"), col("Gender"), col("JobSatisfaction"), col("UndergradMajor"),
      col("EducationTypes"), col("LanguageWorkedWith"), col("ConvertedSalary"),
      col("DevType_segment"),
      // Engineered numeric
      col("Languages_count"), col("EducationTypes_count"),
      col("Salary_num"), col("Salary_log"),
      // Encoded for ML
      col("DevType_segment_indexed"), col("Country_indexed"),
      col("Employment_indexed"), col("Gender_indexed"),
      col("UndergradMajor_indexed"), col("Age_encoded"),
      col("YearsCoding_encoded"), col("JobSatisfaction_encoded")
    )

    dfFinalModel.printSchema()
    dfFinalModel.show(5, truncate = false)

    // Save final transformed data
    println("\n" + "=" * 60)
    println("SAVING FINAL TRANSFORMED DATA")
    println("=" * 60)

    dfFinalModel.write
      .option("header", "true")
      .mode("overwrite")
      .csv("stackoverflow_final_transformed")

    println("Transformed data saved to: stackoverflow_final_transformed/")
    println(s"Rows: ${dfFinalModel.count()}, Cols: ${dfFinalModel.columns.length}")

    // ============================================================
    // 11) SNAPSHOT — Show 20 rows of final dataset
    // ============================================================
    val dfSnapshot = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("stackoverflow_final_transformed")

    dfSnapshot.show(20)

    spark.stop()
  }
}
