import scala.collection.mutable.HashMap
import org.apache.spark.rdd._

Object lsaUtil {

  //Convert docTermFreqs map stored as String on hdfs to map
  def readDocTermfreqs(path: String): RDD[HashMap[String, Int]] = {
    val docTermFreqsMapStr = sc.textFile(path)
    val docTermFreqs = docTermFreqsMapStr.map(l => l.replaceAll("Map", "")).map(l => l.replaceAll("\\(","")).map(l => l.replaceAll("\\)","")).map{
      l => 
      val split = l.split(", ")
      val docTermMap = split.foldLeft(new HashMap[String, Int]()) {
        (map, strMap) =>  { 
          val split2 = strMap.split("->")
          val len = split2.length
          if(len == 2) map += split2(0).trim -> split2(1).trim.toInt else map += split2(0).trim -> 0
          map
        }    
      }
      docTermMap
    }
   docTermFreqs
  }
}
