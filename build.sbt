ThisBuild / version := "1.0.0"
ThisBuild / organization := "com.github.iRoseM"
hisBuild / scalaVersion := "2.13.17"

lazy val root = (project in file("."))
  .settings(
    name := "stackoverflow-survey-analysis",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % "4.1.1" % "compile",
      "org.apache.spark" %% "spark-sql" % "4.1.1" % "compile"    
    )
  )

scalacOptions ++= Seq("-target:17")
