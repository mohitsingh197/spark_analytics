import scala.collection.Map

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Matrix
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.hadoop.io.LongWritable
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation
import org.apache.hadoop.io.Text
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation
import org.apache.spark.mllib.linalg.Matrices

object lsa {
  
   val sc = new SparkContext
   
   //Term Document Matrix weighing scheme. term frequency times inverse document frequency (TF-IDF)
   def termDocWeight(termFrequencyInDoc: Int, totalTermsInDoc: Int, termFreqInCorpus: Int, totalDocs: Int): Double = {
       
       val tf = termFrequencyInDoc.toDouble / totalTermsInDoc
       val docFreq = totalDocs.toDouble / termFreqInCorpus
       val idf = math.log(docFreq)
       tf * idf
   }


  /* 
   *Turn wikipedia xml dump into String XML documents - Starts
   *
   */
  import edu.umd.cloud9.collection.XMLInputFormat
  import org.apache.hadoop.conf.Configuration
  
  val path = "wikipedia/enwiki-20160820-pages-articles-multistream.xml"
  @transient val conf = new Configuration()
  conf.set(XMLInputFormat.START_TAG_KEY, "<page>")
  conf.set(XMLInputFormat.END_TAG_KEY, "</page>")
  
  //Sample this down to a fraction of articles for smaller clusters
  //val kvs = sc.newAPIHadoopFile(path, classOf[XMLInputFormat], classOf[LongWritable], classOf[Text], conf)
  val kvs = sc.newAPIHadoopFile(path, classOf[XMLInputFormat], classOf[LongWritable], classOf[Text], conf).sample(false, 0.001, System.currentTimeMillis().toInt)

  val rawXmls = kvs.map(p => p._2.toString)
  //Turn wikipedia xml dump into String XML documents - Ends

  
  import edu.umd.cloud9.collection.wikipedia._
  /* 
   *Turning wiki XML into plain text document - Starts
   *
   */
  import edu.umd.cloud9.collection.wikipedia.language._
  
  def wikiXmlToPlainText(xml: String): Option[(String, String)] = {
    val page = new EnglishWikipediaPage()
    WikipediaPage.readPage(page, xml)
    if(page.isEmpty || !page.isArticle || page.isRedirect || page.getTitle.contains("disambiguation")) None
    else Some((page.getTitle, page.getContent))
  }

  //Documents keyed by title
  val plainText = rawXmls.flatMap(wikiXmlToPlainText)
  //Turning wiki XML into plain text documents - Ends


  /* 
   *Lemmatization of documents - Starts
   *
   */
  import edu.stanford.nlp.pipeline._
  import java.util.Properties
  import scala.collection.JavaConversions._
  import scala.collection.mutable.ArrayBuffer

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
  //Lemmatization of documents - Ends

  
  /* 
   *Computing TF-IDF - Starts
   *
   */
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
  
  val docIds = docFreqs.map(_._1).zipWithIndex().collectAsMap()

  //Creating TF_IDF weighted vector
  import org.apache.spark.mllib.linalg.Vectors

  val vecs = docTermFreqs.map(termFreqs => {
    val docTotalTerms = termFreqs.values().sum
    val termScores = termFreqs.filter{ case(term, freq) => bTermIds.containsKey(term)}.map{ case(term, freq) => (bTermIds(term), idfs(term) * termFreqs(term) / docTotalTerms)}.toSeq
    Vectors.sparse(bTermIds.size, termScores)
  })

  //Computing TF-IDF - Ends

  
  /* 
   *Find Singular Value Decomposition(SVD) of the Vectors - Starts
   *
   */
  import org.apache.spark.mllib.linalg.distributed.RowMatrix

  vecs.cache()
  val mat = new RowMatrix(vecs)
  val k = 1000
  val svd = mat.computeSVD(k, computeU=true)  
  //Find Singular Value Decomposition(SVD) of the Vectors - Ends

  
  //Find top terms in top concepts by frequency count
  def topTermsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int, numTerms: Int, termIds: Map[Int, String]): Seq[Seq[(String, Double)]] =  {
    val v = svd.V
    val topTerms = new ArrayBuffer[Seq[(String, Double)]]
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

  //Find top documentsin top concepts by frequency count
  def topDocsInTopConcepts(svd: SingularValueDecomposition[RowMatrix, Matrix], numConcepts: Int, numDocs: Int, docIds: Map[Long, String]): Seq[Seq[(String, Double)]] =  {
    val u = svd.U
    val topDocs = new ArrayBuffer[Seq[(String, Double)]]()
    for (i <- 0 until numConcepts) {
      val docWeights = u.rows.map(_.toArray(i)).zipWithUniqueId()
      topDocs += docWeights.top(numDocs).map {
        case(score, id) => (docIds(id), score)
      } 
    }
    topDocs    
  }

  val topConceptTerms = topTermsInTopConcepts(svd, 4, 10, termIds.map(_.swap))
  val topConceptDocs = topDocsInTopConcepts(svd, 4, 10, docIds.map(_.swap))
  for((terms, docs) <- topConceptTerms.zip(topConceptDocs) ) {
    println("Concept terms: " + terms.map(_._1).mkString(", "))
    println("Concept docs: " + docs.map(_._1).mkString(", "))
    println()
  }

  /* 
   * Term -Term Relevance - Starts
   *
   */

   import breeze.linalg.{ DenseMatrix => BDenseMatrix, DenseVector => BDenseVector, SparseVector => BSparseVector }
   
  //Selects a row from a matrix   
  def row(mat: BDenseMatrix[Double], index: Int): Seq[Double] = {
    (0 until mat.cols).map(c => mat(index, c))
  }

  //Selects a row from a matrix.
  def row(mat: Matrix, index: Int): Seq[Double] = {
    val arr = mat.toArray
    (0 until mat.numCols).map(i => arr(index + i * mat.numRows))
  }

  //Selects a row from a distributed matrix.
  def row(mat: RowMatrix, id: Long): Array[Double] = {
    mat.rows.zipWithUniqueId.map(_.swap).lookup(id).head.toArray
  }
  
  //Finds the product of a dense matrix and a diagonal matrix represented by a vector.
  //Breeze doesn't support efficient diagonal representations, so multiply manually.
  def multiplyByDiagonalMatrix(mat: Matrix, diag: Vector[Double]): BDenseMatrix[Double] = {
    val sArr = diag.toArray
    new BDenseMatrix[Double](mat.numRows, mat.numCols, mat.toArray)
      .mapPairs{case ((r, c), v) => v * sArr(c)}
  }

  //Finds the product of a distributed matrix and a diagonal matrix represented by a vector.
  def multiplyByDiagonalMatrix(mat: RowMatrix, diag: Vector[Double]): RowMatrix = {
    val sArr = diag.toArray
    new RowMatrix(mat.rows.map(vec => {
      val vecArr = vec.toArray
      val newArr = (0 until vec.size).toArray.map(i => vecArr(i) * sArr(i))
      Vectors.dense(newArr)
    }))
  }

  //Returns a matrix where each row element is divided by its length.
  def rowsNormalized(mat: BDenseMatrix[Double]): BDenseMatrix[Double] = {
    val newMat = new BDenseMatrix[Double](mat.rows, mat.cols)
    for (r <- 0 until mat.rows) {
      val length = math.sqrt((0 until mat.cols).map(c => mat(r, c) * mat(r, c)).sum)
      (0 until mat.cols).map(c => newMat.update(r, c, mat(r, c) / length))
    }
    newMat
  }

  //Returns a distributed matrix where each row is divided by its length.
  def rowsNormalized(mat: RowMatrix): RowMatrix = {
    new RowMatrix(mat.rows.map(vec => {
      val length = math.sqrt(vec.toArray.map(x => x * x).sum)
      Vectors.dense(vec.toArray.map(_ / length))
    }))
  }

  //Finds terms relevant to a term. Returns the term IDs and scores for the terms with the highest relevance scores to the given term.
  def topTermsForTerm(normalizedVS: BDenseMatrix[Double], termId: Int): Seq[(Double, Int)] = {
    // Look up the row in VS corresponding to the given term ID.
    val termRowVec = new BDenseVector[Double](row(normalizedVS, termId).toArray)

    // Compute scores against every term
    val termScores = (normalizedVS * termRowVec).toArray.zipWithIndex

    // Find the terms with the highest scores
    termScores.sortBy(-_._1).take(10)
  }

  val VS = multiplyByDiagonalMatrix(svd.V, svd.s)
  val normalizedVS = rowsNormalized(VS)
  
  //Finds docs relevant to a doc. Returns the doc IDs and scores for the docs with the highest relevance scores to the given doc.
  def topDocsForDoc(normalizedUS: RowMatrix, docId: Long): Seq[(Double, Long)] = {
    // Look up the row in US corresponding to the given doc ID.
    val docRowArr = row(normalizedUS, docId)
    val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)

    // Compute scores against every doc
    val docScores = normalizedUS.multiply(docRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId

    // Docs can end up with NaN score if their row in U is all zeros.  Filter these out.
    allDocWeights.filter(!_._1.isNaN).top(10)
  }

  //Finds docs relevant to a term. Returns the doc IDs and scores for the docs with the highest relevance scores to the given term.
  def topDocsForTerm(US: RowMatrix, V: Matrix, termId: Int): Seq[(Double, Long)] = {
    val termRowArr = row(V, termId).toArray
    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    // Compute scores against every doc
    val docScores = US.multiply(termRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  def termsToQueryVector(terms: Seq[String], idTerms: Map[String, Int], idfs: Map[String, Double])
    : BSparseVector[Double] = {
    val indices = terms.map(idTerms(_)).toArray
    val values = terms.map(idfs(_)).toArray
    new BSparseVector[Double](indices, values, idTerms.size)
  }

  def topDocsForTermQuery(US: RowMatrix, V: Matrix, query: BSparseVector[Double])
    : Seq[(Double, Long)] = {
    val breezeV = new BDenseMatrix[Double](V.numRows, V.numCols, V.toArray)
    val termRowArr = (breezeV.t * query).toArray

    val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)

    // Compute scores against every doc
    val docScores = US.multiply(termRowVec)

    // Find the docs with the highest scores
    val allDocWeights = docScores.rows.map(_.toArray(0)).zipWithUniqueId
    allDocWeights.top(10)
  }

  def printTopTermsForTerm(normalizedVS: BDenseMatrix[Double],
      term: String, idTerms: Map[String, Int], termIds: Map[Int, String]) {
    printIdWeights(topTermsForTerm(normalizedVS, idTerms(term)), termIds)
  }

  def printTopDocsForDoc(normalizedUS: RowMatrix, doc: String, idDocs: Map[String, Long],
      docIds: Map[Long, String]) {
    printIdWeights(topDocsForDoc(normalizedUS, idDocs(doc)), docIds)
  }

  def printTopDocsForTerm(US: RowMatrix, V: Matrix, term: String, idTerms: Map[String, Int],
      docIds: Map[Long, String]) {
    printIdWeights(topDocsForTerm(US, V, idTerms(term)), docIds)
  }

  def printIdWeights[T](idWeights: Seq[(Double, T)], entityIds: Map[T, String]) {
    println(idWeights.map{case (score, id) => (entityIds(id), score)}.mkString(", "))
  }

  
}

