Object lsa {
   
   //Term Document Matrix weighing scheme. term frequency times inverse document frequency (TF-IDF)
   def termDocWeight(termFrequencyInDoc: Int, totalTermsInDoc: Int, termFreqInCorpus: Int, totalDocs: Int): Double = {
       
       val tf = termFrequencyInDoc.toDouble / totalTermsInDoc
       val docFreq = totalDocs.toDouble / termFreqInCorpus
       val idf = math.log(docFreq)
       tf * idf
   }


  //Turn wikipedia xml dump into String XML documents
  import edu.umd.cloud9.collection.XMLInputFormat
  import org.apache.hadoop.conf.Configuration
  import org.apache.hadoop.io._
  
  val path = "wikipedia/enwiki-20160820-pages-articles-multistream.xml"
  @transient val conf = new Configuration()
  conf.set(XMLInputFormat.START_TAG_KEY, "<page>")
  conf.set(XMLInputFormat.END_TAG_KEY, "</page>")
  
  //Sample this down to a fraction of articles for smaller clusters
  //val kvs = sc.newAPIHadoopFile(path, classOf[XMLInputFormat], classOf[LongWritable], classOf[Text], conf)
  val kvs = sc.newAPIHadoopFile(path, classOf[XMLInputFormat], classOf[LongWritable], classOf[Text], conf).sample(false, 0.01, System.currentTimeMillis().toInt)

  val rawXmls = kvs.map(p => p._2.toString)

  
  //Turning wiki XML into plain text documents
  import edu.umd.cloud9.collection.wikipedia.language._
  import edu.umd.cloud9.collection.wikipedia._
  
  def wikiXmlToPlainText(xml: String): Option[(String, String)] = {
    val page = new EnglishWikipediaPage()
    WikipediaPage.readPage(page, xml)
    if(page.isEmpty || !page.isArticle || page.isRedirect || page.getTitle.contains("disambiguation")) None
    else Some((page.getTitle, page.getContent))
  }

  //Documents keyed by title
  val plainText = rawXmls.flatMap(wikiXmlToPlainText)


  //Lemmatization of documents
  import edu.stanford.nlp.pipeline._
  import edu.stanford.nlp.ling.CoreAnnotations._
  import java.util.Properties
  import scala.collection.mutable.ArrayBuffer
  import scala.collection.JavaConversions._

  def createNLPPipeline(): StanfordCoreNLP = {
    val props = new Properties()
    props.put("annotators","tokenize, ssplit, pos, lemma")
    new StanfordCoreNLP(props)
  }

  def isOnlyLetters(str: String): Boolean = {
    str.forall(c => Character.isLetter(c))
  }

  def plainTextToLemmas(text: String, stopWords: Set[String], pipeline: StanfordCoreNLP): Seq[String] = {
    val doc = new Annotation(text)
    pipeline.annotate(doc)

    val lemmas = new ArrayBuffer[String]()
    val sentences = doc.get(classOf[SentencesAnnotation])
    for (sentence <- sentences;
        token <- sentence.get(classOf[TokensAnnotation])) {
        
        val lemma = token.get(classOf[LemmaAnnotation])
        if(lemma.length > 2 && !stopWords.contains(lemma) && isOnlyLetters(lemma)) {
          lemmas += lemma.toLowerCase
        }
    }
    lemmas
  }

  val stopwords = sc.broadcast(scala.io.Source.fromFile("/media/mohit/Rian/Data/wikipedia/stopwords.txt").getLines().toSet).value

  val lemmatized = plainText.mapPartitions( it => {
    val pipeline = createNLPPipeline()
    it.map { case(title, contents) => plainTextToLemmas(contents, stopwords, pipeline)
    }
  })

  
  //Computing TF-IDF
  import scala.collection.mutable.HashMap

  val docTermFreqs = lemmatized.map(terms => {
    val termFreqs = terms.foldLeft(new HashMap[String, Int]()) {
      (map, term) => {
        map += term -> (map.getOrElse(term, 0) + 1)
        map
      } 
    }
    termFreqs
  })

  //Optionally save this to hdfs to avoid recomputing if you want to discontinue now and resume later
  //sc.saveAsTextFile("docTermFreqs")

  //Read the saved docTermFreqs from hdfs
  //val docTermFreqs = readDocTermfreqs("docTermFreqs")

  docTermFreqs.cache()

  //calculating doc frequencies
  val docFreqs = docTermFreqs.flatMap(_.keySet).map((_, 1)).reduceByKey(_+_)

  //Keep to N most frequent terms
  /*
  val numTerms = 50000
  val ordering = Ordering.by[(String,Int),Int](_._2)
  val topDocFreqs = docFreqs.top(numTerms)(ordering)
   */

  //Calculate number of docs
  val numDocs = docTermFreqs.count

  //calculating inverse document frequencies
  val idfs = docFreqs.map { case(term, count) => (term, math.log(numDocs.toDouble / count))}.collectAsMap
  
  //Assigning Ids to the String terms
  val termIds = idfs.keys.zipWithIndex.toMap
  val bTermIds = sc.broadcast(termIds).value

  //Creating TF_IDF weighted vector
  import org.apache.spark.mllib.linalg.Vectors

  val vecs = docTermFreqs.map(termFreqs => {
    val docTotalTerms = termFreqs.values().sum
    val termScores = termFreqs.filter{ case(term, freq) => bTermIds.containsKey(term)}.map{ case(term, freq) => (bTermIds(term), idfs(term) * termFreqs(term) / docTotalTerms)}.toSeq
    Vectors.sparse(bTermIds.size, termScores)
  })

  
  //Find Singular Value Decomposition(SVD) of the Vectors
  import org.apache.spark.mllib.linalg.distributed.RowMatrix

  vecs.cache()
  val mat = new RowMatrix(vecs)
  val k = 1000
  val svd = mat.computeSVD(k, computeU=true)  

  
  //Find top terms in top concepts
  def topTermsInTopConcepts(svd: SingularValueDecomposition, numConcepts: Int, numTerms: Int, termIds: Map[Long, String]): Seq[Seq[String, Double]] =  {
    val v = svd.V
    val topTerms = new ArrayBuffer[Seq[String, Double]]()
    val arr = v.toArray
    for (i <- 0 until numConcepts) {
      val offs = i * v.numRows
      val termWeights = arr.slice(offs, offs + v.numRows).zipWithIndex
      val sorted = termWeights.sortBy(_._1)
      topTerms += sorted.take(numTerms).map {
        case(score, id) => (termIds(id), score)
      }
    }
    topTerms
  }

  //Find top documentsin top concepts
  def topDocsInTopConcepts(svd: SingularValueDecomposition, numConcepts: Int, numDocs: Int, docIds: Map[Long, String]): Seq[Seq[String, Double]] =  {
    val u = svd.U
    val topDocs = new ArrayBuffer[Seq[String, Double]]()
    for (i <- 0 until numConcepts) {
      val docWeights = u.rows.map(_.toArray(i)).zipWithUniqueId()
      topDocs += docWeights.top(numDocs).map {
        case(score, id) => (docIds(id), score)
      } 
    }
    topDocs    
  }

  val topConceptTerms = topTermsInTopConcepts(svd, 4, 10, termIds)
  val topConceptDocs = topDocsInTopConcepts(svd, 4, 10, termIds)
  for((terms, docs) <- topConceptTerms.zip(topConceptDocs) ) {
    println("Concept terms: " + terms.map(_._1).mkString(", "))
    println("Concept docs: " + docs.map(_._1)).mkString(", "))
    println()
  }

}

