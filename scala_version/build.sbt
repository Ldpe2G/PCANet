lazy val commonSettings = Seq(
  version := "0.1.0",
  scalaVersion := "2.11.8"
)

lazy val root = (project in file(".")).
  settings(commonSettings: _*).
  settings(
    name := "PCANet_Scala"
  )


libraryDependencies  ++= Seq( 
  "nu.pattern" % "opencv" % "2.4.9-7",
  "commons-io" % "commons-io" % "2.5"
)
