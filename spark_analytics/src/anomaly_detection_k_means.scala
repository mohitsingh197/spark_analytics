
val sc  = new SparkContext();

val rawData = sc.textFile("anomaly_detection_kmeans/kddcup.data")

//counts the labels
val labels_count = rawData.map(l => (l.split(",").last,1)).reduceByKey(_ + _)

//removes text categorical columns
import org.apache.spark.mllib.linalg._

val labelsAndData = rawData.map { line =>
  val buffer = line.split(',').toBuffer //converts to Buffer, a mutable list
  buffer.remove(1, 3)
  val label = buffer.remove(buffer.length-1)
  val vector = Vectors.dense(buffer.map(_.toDouble).toArray)
  (label,vector)
}

val data = labelsAndData.values.cache()


//Initialize and run Kmeans
import org.apache.spark.mllib.clustering._

val kmeans = new KMeans()
val model = kmeans.run(data)

model.clusterCenters.foreach(println) //prints 2 clusters. Not enough!!


//uses the model to assign each data point to a cluster, counts occurrences of cluster and label pairs, and prints them
val clusterLabelCount = labelsAndData.map { case (label,datum) =>
  val cluster = model.predict(datum)
  (cluster,label)
}.countByValue

clusterLabelCount.toSeq.sorted.foreach {
  case ((cluster,label),count) =>
    println(f"$cluster%1s$label%18s$count%8s")
}


// Euclidean distance function
def distance(a: Vector, b: Vector) =
  math.sqrt(a.toArray.zip(b.toArray).
    map(p => p._1 - p._2).map(d => d * d).sum)

//function that returns the distance from a data point to its nearest cluster’s centroid
def distToCentroid(datum: Vector, model: KMeansModel) = {
  val cluster = model.predict(datum)
  val centroid = model.clusterCenters(cluster)
  distance(centroid, datum)
}


//it’s possible to define a function that measures the average distance to centroid, for a model built with a given k:
import org.apache.spark.rdd._

def clusteringScore(data: RDD[Vector], k: Int) = {
  val kmeans = new KMeans()
  kmeans.setK(k)
  val model = kmeans.run(data)
  data.map(datum => distToCentroid(datum, model)).mean()
}

//Now, this can be used to evaluate values of k from, say, 5 to 40:
(5 to 40 by 5).map(k => (k, clusteringScore(data, k))).foreach(println)


//Best valueof k seems to be past 100. Running Kmeans for k=100
val k = 100
kmeans.setK(k)
val model100 = kmeans.run(data)
println(model100.clusterCentres)
