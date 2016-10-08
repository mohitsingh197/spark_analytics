import org.apache.spark.SparkContext


object audio {
  
  val sc = new SparkContext()
  
  
  //read user-artist-count data
  val rawUserArtistData = sc.textFile("audio_scrobble/user_artist_data.txt")
  val stats_userIds = rawUserArtistData.map(_.split(' ')(0).toInt).stats
  val stats_artistIds = rawUserArtistData.map(_.split(' ')(1).toInt).stats
  
  
  //parse artist id/name
  val rawArtistData = sc.textFile("audio_scrobble/artist_data.txt")
  val artistById = rawArtistData.flatMap { line => 
    val (id, name) = line.span(_ != '\t')
    if (name.isEmpty()) {
      None
    } else {
      try {
        Some((id.toInt, name.trim()))
      } catch {
        case e: NumberFormatException => None
      }
    }
  }
  
  
  //parse artist aliases
  val rawArtistAlias = sc.textFile("audio_scrobble/artist_alias.txt")
  val artistAlias = rawArtistAlias.flatMap { line => 
    val tokens = line.split('\t')
    if(tokens(0).isEmpty()) {
      None
    } else {
      Some((tokens(0).toInt, tokens(1).toInt))
    }
      
  }.collectAsMap
  artistById.lookup(6803336).head
  artistById.lookup(1000010).head
  
  
  //broadcast artist id-name map
  import org.apache.spark.mllib.recommendation._
  val bArtistAlias = sc.broadcast(artistAlias)
  
  
  //train
  val trainData = rawUserArtistData.map { line => 
    val Array(userId, artistId, count) = line.split(' ').map { _.toInt }
    val finalArtistId = bArtistAlias.value.getOrElse(artistId, artistId)
    Rating(userId, finalArtistId, count)
  }.cache()
  val model = ALS.trainImplicit(trainData, 10, 20, 0.01, 50.0)
  model.userFeatures.mapValues { _.mkString(", ") }.first()
  model.productFeatures.mapValues { _.mkString(", ") }.first()

  
  //recommend
  val recommendations = model.recommendProducts(2093760, 5)
  val recoArtistIds = recommendations.map { r => r.product }.toSet
  val recoArtistsName = artistById.filter{ case (id, _) => recoArtistIds.contains(id) }.map{ case (_, name) => name }.collect()
  
  
}