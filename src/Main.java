import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import java.util.Properties;
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello World!");
        SparkSession spark = SparkSession
                .builder()
                .config("SPARK_LOCAL_IP", "127.0.0.1")
                .config("executor-memory", "2g")
                .master("local[*]")
                .getOrCreate();
        
        //获取数据，得到一个dataset ( features,model)
        spark.read().csv("file:///Users/liuxiao/Downloads/test.csv").createOrReplaceTempView("test");
        spark.sql("select `_c0` as word,collect_list(`_c0`) as features from test group by `_c0`").createOrReplaceTempView("vectors");
        spark.read().csv("file:///Users/liuxiao/Downloads/test_1.csv").createOrReplaceTempView("test_1");
        spark.sql("select `_c0` as model,`_c1` as word from test_1 ").createOrReplaceTempView("models");
        Dataset<Row> ds = spark.sql("select a.features as features,b.model  as model from vectors a join models b on a.word = b.word");


        // word2vec
        Word2Vec word2Vec = new Word2Vec().setInputCol("features").setOutputCol("result").setWindowSize(10).setVectorSize(5).setMinCount(0);
        Word2VecModel model = word2Vec.fit(ds);
        Dataset<Row> result = model.transform(ds);

        //kmeans
        KMeans kmeans = new KMeans().setK(5).setSeed(1L).setFeaturesCol("result");
        KMeansModel kMeansModel = kmeans.fit(result);
        Dataset<Row> clusters = kMeansModel.transform(result).select("model", "prediction");

        //将clusters 这个 dataset写入数据库
        Properties connectionProperties = new Properties();
        connectionProperties.put("user", "liuxiao");
        connectionProperties.put("password", "");
        clusters.write().mode("overwrite").jdbc("jdbc:postgresql://127.0.0.1:5432/postgres", "public.cluster_result", connectionProperties);
    }
}