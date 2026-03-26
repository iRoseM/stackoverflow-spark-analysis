import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object SQL_Analysis {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("StackOverflow SQL Analysis")
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // Load the preprocessed data
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("stackoverflow_final_transformed.csv")
    
    println("=== Dataset Loaded for SQL Analysis ===")
    println(s"Rows: ${df.count()} | Cols: ${df.columns.length}")
    println()

    // Register as temporary view
    df.createOrReplaceTempView("developer_survey")
    

    // ============================================================
    // QUERY 1: Average Salary by Developer Type
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 1: Average Salary by Developer Type")
    println("=" * 60)
    
    val query1 = """
  SELECT 
    DevType_segment,
    COUNT(*) as developer_count,
    ROUND(AVG(Salary_num), 2) as avg_salary_usd,
    ROUND(STDDEV(Salary_num), 2) as salary_stddev,     
    ROUND(VARIANCE(Salary_num), 2) as salary_variance   
  FROM developer_survey
  GROUP BY DevType_segment
  ORDER BY avg_salary_usd DESC
"""
    
    spark.sql(query1).show(10, truncate = false)
    
    // ============================================================
    // QUERY 2: Statistical Summary of Salaries
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 2: Statistical Summary of Salaries")
    println("=" * 60)
    
    val query2 = """
  SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT Country) as distinct_countries,
    COUNT(DISTINCT DevType_segment) as distinct_roles,
    ROUND(AVG(Salary_num), 0) as mean_salary,
    ROUND(STDDEV(Salary_num), 0) as stddev_salary,
    ROUND(VARIANCE(Salary_num), 0) as variance_salary,
    ROUND(PERCENTILE(Salary_num, 0.25), 0) as q1_salary,
    ROUND(PERCENTILE(Salary_num, 0.5), 0) as median_salary,
    ROUND(PERCENTILE(Salary_num, 0.75), 0) as q3_salary,
    ROUND(MIN(Salary_num), 0) as min_salary,
    ROUND(MAX(Salary_num), 0) as max_salary
  FROM developer_survey
  WHERE Salary_num > 0 AND Salary_num < 500000
"""
    
    spark.sql(query2).show(truncate = false)
    
    // ============================================================
    // QUERY 3: Educational Pathways to Data Roles
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 3: Educational Pathways to Data Roles")
    println("=" * 60)
    
    val query3 = """
 SELECT 
  UndergradMajor,
  COUNT(*) as total_developers,
  SUM(CASE WHEN DevType_segment = 'Data' THEN 1 ELSE 0 END) as data_roles_count,
  ROUND(SUM(CASE WHEN DevType_segment = 'Data' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_in_data_roles
FROM developer_survey
WHERE UndergradMajor IS NOT NULL
GROUP BY UndergradMajor
HAVING total_developers >= 30
ORDER BY pct_in_data_roles DESC
LIMIT 10
    """
    
    spark.sql(query3).show(truncate = false)
    
    // ============================================================
    // QUERY 4: Self-Learning vs Formal Education Impact
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 4: Self-Learning vs Formal Education Impact")
    println("=" * 60)
    
    val query4 = """
      SELECT 
        CASE 
          WHEN EducationTypes LIKE '%Taught yourself%' 
               AND EducationTypes NOT LIKE '%Bachelor%' 
               AND EducationTypes NOT LIKE '%Master%'
            THEN 'Self-Learning Only'
          WHEN (EducationTypes LIKE '%Bachelor%' OR EducationTypes LIKE '%Master%') 
               AND EducationTypes LIKE '%Taught yourself%'
            THEN 'Formal + Self-Learning'
          WHEN (EducationTypes LIKE '%Bachelor%' OR EducationTypes LIKE '%Master%') 
            THEN 'Formal Only'
          ELSE 'Mixed/Other'
        END as learning_path,
        COUNT(*) as developer_count,
        ROUND(AVG(Salary_num), 0) as avg_salary
      FROM developer_survey
      WHERE Salary_num > 0
      GROUP BY learning_path
      ORDER BY avg_salary DESC
    """
    
    spark.sql(query4).show(truncate = false)
    
    // ============================================================
    // QUERY 5: Top Programming Languages by Developer Type
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 5: Top Programming Languages per Developer Type")
    println("=" * 60)
    
    val query5 = """
      WITH language_counts AS (
        SELECT 
          DevType_segment,
          language,
          COUNT(*) as lang_count
        FROM (
          SELECT 
            DevType_segment,
            EXPLODE(SPLIT(LanguageWorkedWith, ';')) as language
          FROM developer_survey
        ) t
        GROUP BY DevType_segment, language
      ),
      ranked AS (
        SELECT 
          DevType_segment,
          language,
          lang_count,
          ROW_NUMBER() OVER (PARTITION BY DevType_segment ORDER BY lang_count DESC) as rank
        FROM language_counts
      )
      SELECT DevType_segment, language, lang_count
      FROM ranked
      WHERE rank <= 3
      ORDER BY DevType_segment, rank
    """
    
    spark.sql(query5).show(20, truncate = false)
    
    // ============================================================
    // QUERY 6: Salary Distribution by Age and Experience
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 6: Salary Distribution by Age and Experience Groups")
    println("=" * 60)
    
    val query6 = """
      SELECT 
        CASE 
          WHEN Age LIKE 'Under%' THEN 'Under 18 (Pre-Career)'
          WHEN Age LIKE '18 -%' THEN '18-24 (Early Career)'
          WHEN Age LIKE '25 -%' THEN '25-34 (Mid Career)'
          WHEN Age LIKE '35 -%' THEN '35-44 (Established)'
          WHEN Age LIKE '45 -%' THEN '45-54 (Senior)'
          WHEN Age LIKE '55 -%' THEN '55-64 (Veteran)'
          WHEN Age LIKE '65%' THEN '65+ (Retirement)'
          ELSE Age
        END as age_group,
        CASE 
          WHEN YearsCoding LIKE '0-2%' THEN '0-2 yrs (Junior)'
          WHEN YearsCoding LIKE '3-5%' THEN '3-5 yrs (Intermediate)'
          WHEN YearsCoding LIKE '6-8%' THEN '6-8 yrs (Mid-Level)'
          WHEN YearsCoding LIKE '9-11%' THEN '9-11 yrs (Senior)'
          WHEN YearsCoding LIKE '12-14%' THEN '12-14 yrs (Expert)'
          WHEN YearsCoding LIKE '15-17%' THEN '15-17 yrs (Veteran)'
          WHEN YearsCoding LIKE '18-20%' THEN '18-20 yrs (Master)'
          WHEN YearsCoding LIKE '21-23%' THEN '21-23 yrs (Legend)'
          WHEN YearsCoding LIKE '24-26%' THEN '24-26 yrs (Icon)'
          WHEN YearsCoding LIKE '27-29%' THEN '27-29 yrs (Pioneer)'
          WHEN YearsCoding LIKE '30 or%' THEN '30+ yrs (Oracle)'
          ELSE YearsCoding
        END as experience_level,
        COUNT(*) as count,
        ROUND(AVG(Salary_num), 0) as avg_salary_usd,
        ROUND(PERCENTILE(Salary_num, 0.5), 0) as median_salary_usd
      FROM developer_survey
      WHERE Salary_num > 0 AND Salary_num < 500000
        AND Age IS NOT NULL
        AND YearsCoding IS NOT NULL
      GROUP BY age_group, experience_level
      HAVING count >= 10
      ORDER BY 
        CASE age_group
          WHEN 'Under 18 (Pre-Career)' THEN 1
          WHEN '18-24 (Early Career)' THEN 2
          WHEN '25-34 (Mid Career)' THEN 3
          WHEN '35-44 (Established)' THEN 4
          WHEN '45-54 (Senior)' THEN 5
          WHEN '55-64 (Veteran)' THEN 6
          WHEN '65+ (Retirement)' THEN 7
          ELSE 8
        END,
        CASE 
          WHEN experience_level LIKE '0-2%' THEN 1
          WHEN experience_level LIKE '3-5%' THEN 2
          WHEN experience_level LIKE '6-8%' THEN 3
          WHEN experience_level LIKE '9-11%' THEN 4
          WHEN experience_level LIKE '12-14%' THEN 5
          WHEN experience_level LIKE '15-17%' THEN 6
          WHEN experience_level LIKE '18-20%' THEN 7
          WHEN experience_level LIKE '21-23%' THEN 8
          WHEN experience_level LIKE '24-26%' THEN 9
          WHEN experience_level LIKE '27-29%' THEN 10
          WHEN experience_level LIKE '30+%' THEN 11
          ELSE 12
        END
    """
    
    spark.sql(query6).show(200, truncate = false)
        
    // ============================================================
    // QUERY 7: Global Talent Distribution
    // ============================================================
    println("\n" + "=" * 60)
    println("QUERY 7: Global Talent Distribution - Top 15 Countries")
    println("=" * 60)
    
    val query7 = """
SELECT 
  Country,
  COUNT(*) as developer_count,
  ROUND(AVG(Salary_num), 0) as avg_salary_usd,
  SUM(CASE WHEN DevType_segment = 'Data' THEN 1 ELSE 0 END) as data_roles_count,
  ROUND(SUM(CASE WHEN DevType_segment = 'Data' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) as pct_data_roles
FROM developer_survey
WHERE Country IS NOT NULL AND Salary_num > 0
GROUP BY Country
HAVING developer_count >= 50
ORDER BY developer_count DESC
LIMIT 15
    """
    
    spark.sql(query7).show(truncate = false)
        
    spark.stop()
  }
}