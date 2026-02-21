ThisBuild / version := "1.0.0"
ThisBuild / organization := "com.github.iRoseM"
ThisBuild / scalaVersion := "2.13.17"

lazy val root = (project in file("."))
  .settings(
    name := "stackoverflow-survey-analysis",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "4.1.1",
      "org.apache.spark" %% "spark-sql" % "4.1.1"
    )
  )

scalacOptions ++= Seq("-target:17")
