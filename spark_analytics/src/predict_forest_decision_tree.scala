import org.apache.spark.SparkContext

object predict_forest_decision_tree {

  val sc = new SparkContext

  import org.apache.spark.mllib.linalg._
  import org.apache.spark.mllib.regression._

  val rawData = sc.textFile("decision_tree/covtype.data.gz")

  val data = rawData.map { line =>
    val values = line.split(',').map(_.toDouble)
    val featureVector = Vectors.dense(values.init)
    val label = values.last - 1
    LabeledPoint(label, featureVector)
  }

  val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
  trainData.cache()
  cvData.cache()
  testData.cache()

  import org.apache.spark.mllib.evaluation._
  import org.apache.spark.mllib.tree._
  import org.apache.spark.mllib.tree.model._
  import org.apache.spark.rdd._

  def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]): MulticlassMetrics = {
    val predictionsAndLabels = data.map(example => (model.predict(example.features), example.label))
    new MulticlassMetrics(predictionsAndLabels)
  }

  //train
  val model = DecisionTree.trainClassifier(trainData, 7, Map[Int, Int](), "gini", 4, 100)

  //Cross validation
  val metrics = getMetrics(model, cvData)
  
  //F1 Score
//  Range(0, 6).map { cat => (metrics.precision(cat), metrics.recall(cat)) }
  val precRecall = (0 until 7).map { cat => (metrics.precision(cat), metrics.recall(cat)) }
  
  def computeF1Score(precision: Double, recall: Double): Double = {
    2 * (precision * recall) / (precision + recall)  
  }
  
  val f1Scores = precRecall.map(l => computeF1Score(l._1, l._2))
  
  
  //cALCULATE IMPURITY
  import org.apache.spark.rdd._

  def classProbabilities(data: RDD[LabeledPoint]): Array[Double] = {
    val countsByCategory = data.map(_.label).countByValue()
    val counts = countsByCategory.toArray.sortBy(_._1).map(_._2)
    counts.map(_.toDouble / counts.sum)
  }
  
  val trainPriorProbabilities = classProbabilities(trainData)
  val cvPriorProbabilities = classProbabilities(cvData)
  trainPriorProbabilities.zip(cvPriorProbabilities).map {
    case (trainProb, cvProb) => trainProb * cvProb
  }.sum
  
  
  //Tuning
  val evaluations =
  for (impurity <- Array("gini", "entropy");
       depth    <- Array(1, 20);
       bins     <- Array(10, 300))
    yield {
      val model = DecisionTree.trainClassifier(
        trainData, 7, Map[Int,Int](), impurity, depth, bins)
      val predictionsAndLabels = cvData.map(example =>
        (model.predict(example.features), example.label)
      )
      val accuracy =
        new MulticlassMetrics(predictionsAndLabels).precision
      ((impurity, depth, bins), accuracy)
    }

  evaluations.sortBy(_._2).reverse.foreach(println)
  
  //It turns out that entropy with 20 depth and 300 bins is the best with 91% accuracy ((entropy,20,300),0.9125701825174705)
  //Train again including CV set
  val modelBest = DecisionTree.trainClassifier(trainData.union(cvData), 7, Map[Int,Int](), "entropy", 20, 300)
  //Test set
  val metricsTest = getMetrics(modelBest, testData)
  metricsTest.accuracy

  

}