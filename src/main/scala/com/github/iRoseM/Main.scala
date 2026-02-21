package com.github.iRoseM

import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    // 1. إنشاء جلسة Spark
    val spark = SparkSession.builder()
      .appName("StackOverflow Survey Analysis")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // 2. تحديد مسار ملف CSV
    val csvFilePath = "data/survey_results_public.csv"

    // 3. قراءة ملف CSV
    println(s"📂 قراءة الملف من: $csvFilePath")
    val df = spark.read
      .option("header", "true")        // السطر الأول أسماء الأعمدة
      .option("inferSchema", "true")    // استنتاج نوع البيانات
      .csv(csvFilePath)

    // 4. عرض معلومات عن البيانات
    println(s"✅ تم تحميل ${df.count()} سطر")
    
    println("\n📋 --- أول 5 أسطر ---")
    df.show(5, truncate = false)
    
    println("\n📊 --- مخطط البيانات (Schema) ---")
    df.printSchema()
    
    println("\n📈 --- إحصائيات سريعة ---")
    df.describe().show()

    // 5. إيقاف جلسة Spark
    spark.stop()
  }
}