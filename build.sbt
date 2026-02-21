ThisBuild / version := "1.0.0"
ThisBuild / organization := "com.example"
ThisBuild / scalaVersion := "2.13.17"

lazy val root = (project in file("."))
  .settings(
    name := "stackoverflow-survey-analysis",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "4.1.1" % "provided",
      "org.apache.spark" %% "spark-sql" % "4.1.1" % "provided"
    )
  )

scalacOptions ++= Seq("-target:17")