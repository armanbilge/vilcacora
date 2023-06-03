ThisBuild / tlBaseVersion := "0.0"

ThisBuild / organization := "com.armanbilge"
ThisBuild / organizationName := "Arman Bilge"
ThisBuild / developers ++= List(
  tlGitHubDev("archisman-dey", "Archisman Dey"),
  tlGitHubDev("armanbilge", "Arman Bilge"),
  tlGitHubDev("valencik", "Andrew Valencik"),
)
ThisBuild / startYear := Some(2023)
ThisBuild / tlSonatypeUseLegacyHost := false

ThisBuild / crossScalaVersions := Seq("3.3.0", "2.13.10")

ThisBuild / githubWorkflowJavaVersions := Seq(JavaSpec.temurin("17"))
ThisBuild / githubWorkflowBuildPreamble +=
  WorkflowStep.Run(List("/home/linuxbrew/.linuxbrew/bin/brew update"), name = Some("brew update"))
ThisBuild / githubWorkflowBuildPreamble ++= nativeBrewInstallWorkflowSteps.value

val CatsEffectVersion = "3.5.0"

lazy val root = tlCrossRootProject.aggregate(ir, onnx, runtime)

lazy val ir = project
  .enablePlugins(ScalaNativePlugin)
  .settings(
    name := "vilcacora-ir",
  )

lazy val downloadOnnxProto = taskKey[File]("Download ONNX proto")

lazy val onnx = project
  .enablePlugins(ScalaNativePlugin)
  .settings(
    name := "vilcacora-onnx",
    libraryDependencies ++= Seq(
      "com.thesamet.scalapb" %% "scalapb-runtime" % scalapb.compiler.Version.scalapbVersion % "protobuf",
    ),
    Compile / PB.generate := (Compile / PB.generate).dependsOn(Compile / downloadOnnxProto).value,
    Compile / PB.targets := Seq(
      scalapb.gen() -> (Compile / sourceManaged).value / "scalapb",
    ),
    Compile / downloadOnnxProto := {
      import sbt.util.CacheImplicits._
      (Compile / downloadOnnxProto).previous(sbt.singleFileJsonFormatter).getOrElse {
        streams.value.log.info("Downloading onnx.proto3 ...")
        val f = target.value / "protobuf_external_src" / "onnx.proto"
        IO.transfer(
          url("https://raw.githubusercontent.com/onnx/onnx/v1.14.0/onnx/onnx.proto3").openStream(),
          f,
        )
        f
      }
    },
  )
  .dependsOn(ir)

lazy val runtime = project
  .enablePlugins(ScalaNativePlugin, ScalaNativeBrewedConfigPlugin)
  .dependsOn(ir)
  .settings(
    name := "vilcacora-runtime",
    libraryDependencies ++= Seq(
      "org.typelevel" %%% "cats-effect-kernel" % CatsEffectVersion,
    ),
    nativeBrewFormulas ++= Set("openblas", "mlpack"),
  )
