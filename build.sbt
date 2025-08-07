ThisBuild / tlBaseVersion := "0.0"

ThisBuild / organization := "com.armanbilge"
ThisBuild / organizationName := "Arman Bilge"
ThisBuild / developers ++= List(
  tlGitHubDev("pantShrey", "Shrey Pant"),
  tlGitHubDev("armanbilge", "Arman Bilge"),
  tlGitHubDev("valencik", "Andrew Valencik"),
)
ThisBuild / startYear := Some(2023)
ThisBuild / tlSonatypeUseLegacyHost := false

ThisBuild / crossScalaVersions := Seq("3.3.4", "2.13.16")

ThisBuild / githubWorkflowJavaVersions := Seq(JavaSpec.temurin("21"))
ThisBuild / githubWorkflowBuildPreamble +=
  WorkflowStep.Run(List("/home/linuxbrew/.linuxbrew/bin/brew update"), name = Some("brew update"))
ThisBuild / githubWorkflowBuildPreamble ++= nativeBrewInstallWorkflowSteps.value

ThisBuild / Test / testOptions += Tests.Argument("+l") // for munit logging

val CatsEffectVersion = "3.7.0-RC1"
val CatsVersion = "2.12.0"
val MunitVersion = "1.0.4"
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
      "org.typelevel" %%% "cats-core" % CatsVersion,
      "org.scalameta" %%% "munit" % MunitVersion % Test,
    ),
    nativeConfig ~= { _.withEmbedResources(true) },
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
          url("https://raw.githubusercontent.com/onnx/onnx/rel-1.9.0/onnx/onnx.proto3")
            .openStream(),
          f,
        )
        f
      }
    },
    scalacOptions += "-Wconf:src=src_managed/.*:i",
  )
  .dependsOn(ir)

lazy val runtime = project
  .enablePlugins(ScalaNativePlugin, ScalaNativeBrewedConfigPlugin)
  .dependsOn(ir)
  .settings(
    name := "vilcacora-runtime",
    libraryDependencies ++= Seq(
      "org.typelevel" %%% "cats-effect" % CatsEffectVersion,
      "org.typelevel" %%% "cats-core" % CatsVersion,
      "org.scalameta" %%% "munit" % MunitVersion % Test,
    ),
    nativeBrewFormulas ++= Set("openblas", "mlpack", "libsvm"),
  )
