import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.List;
import java.util.Properties;

public class Main {
    public static void main(String[] args) {
        System.out.println("Hello World!");
        Properties connectionProperties = new Properties();
        connectionProperties.put("user", "liuxiao");
        connectionProperties.put("password", "");

        SparkSession spark = SparkSession
                .builder()
                .config("SPARK_LOCAL_IP", "127.0.0.1")
                .master("local[*]")
                .getOrCreate();
        
        //获取数据，得到一个dataset ( features,model)
        /*
        spark.read().csv("file:///Users/liuxiao/word2vec/src/look1.csv").withColumnRenamed("_c0","line").createOrReplaceTempView("temp");
        Dataset<Row> ds = spark.sql("select split(line,' ') as features,split(line,' ')[0] as model from temp");*/

        spark.read().jdbc("jdbc:postgresql://127.0.0.1:5432/postgres","(select features,features[1] as model from (select regexp_split_to_array(words,' ') as features from look1) as a) as dt",connectionProperties).printSchema();
        Dataset<Row> ds = spark.read().jdbc("jdbc:postgresql://127.0.0.1:5432/postgres","(select features,features[1] as model from (select regexp_split_to_array(words,' ') as features from look1) as a) as dt",connectionProperties);

        // word2vec
        Word2Vec word2Vec = new Word2Vec().setInputCol("features").setOutputCol("result").setWindowSize(10).setVectorSize(5).setMinCount(0);
        Word2VecModel model = word2Vec.fit(ds);
        Dataset<Row> result = model.getVectors();
        //for (int i = 0; i < wtf.size() ; i++) {
        //    System.out.println(wtf.get(i));
        //}
        //Dataset<Row> result = model.transform(ds);

        //kmeans

        KMeans kmeans = new KMeans().setK(5).setSeed(1L).setFeaturesCol("vector");
        KMeansModel kMeansModel = kmeans.fit(result);
        Dataset<Row> clusters;
        clusters = kMeansModel.transform(result).select("word", "prediction");
        clusters.printSchema();

        //将clusters 这个 dataset写入数据库

       clusters.write().mode("overwrite").jdbc("jdbc:postgresql://127.0.0.1:5432/postgres", "public.cluster_result", connectionProperties);
    }
}