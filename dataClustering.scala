import org.apache.spark.sql.SparkSession
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()
import org.apache.spark.ml.clustering.Kmeans

val dataset = spark.read.option("header","true").option("inferSchema","true").csv("Wholesale customers data.csv")
val featureData = dataset.select($"Fresh", $"Milk", $"Grocery", $"Frozen", $"Detergents_Paper", $"Delicassen")

import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

val trainData = assembler.transform(featureData).select("features")
val kMeans = new KMeans().setK(3).setSeed(1)
val model = kmeans.fit(trainData)

val SSE = model.computeCost(trainData)
println(s"SSE is $SSE")

println("Clusters -- ")
model.clusterCenters.foreach(println)
