object test {;import org.scalaide.worksheet.runtime.library.WorksheetSupport._; def main(args: Array[String])=$execute{;$skip(57); 
  println("Welcome to the Scala worksheet");$skip(37); 
  
  
  val values = Array(1.0, 2.0);System.out.println("""values  : Array[Double] = """ + $show(values ));$skip(14); val res$0 = 
  values.init;System.out.println("""res0: Array[Double] = """ + $show(res$0));$skip(14); val res$1 = 
  values.last;System.out.println("""res1: Double = """ + $show(res$1))}
  
  }
