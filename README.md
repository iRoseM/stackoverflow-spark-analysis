# Stack Overflow 2018 Developer Survey Analysis
Big data analytics project using Apache Spark (Scala) to analyze developer roles, salaries, and career patterns from the Stack Overflow 2018 Developer Survey (~100K responses).

## Authors
- Rose Mady	444200107
- Layan Alfawzan	444200793
- Aljawharah Alwabel	444200750
- Layan Aldbays	444200653

## Results Summary

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | 41.6% | 33.7% |
| Random Forest | 52.4% | 37.5% |

- Highest salaries: "Other" roles ($85.6K) and Data roles ($84.0K)
- Top predictors: Employment status, languages count, undergraduate major

## Dataset

- Original: 98,855 records, 129 columns
- After preprocessing: 81,582 records, 24 columns
- Target classes: Web (68%), Student (18%), Other (6%), Mobile (5%), Data (3%)


## Requirements

- Apache Spark 4.0+
- Scala 2.13+
- Dataset: [Stack Overflow 2018 Developer Survey (Kaggle)](https://www.kaggle.com/datasets/stackoverflow/stack-overflow-2018-developer-survey)
