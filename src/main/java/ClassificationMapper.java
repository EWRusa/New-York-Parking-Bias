import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.monotonically_increasing_id;

public final class ClassificationMapper {
    private static Dataset<Row> dataset;

    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("ClassificationMapper").master("local")
                .getOrCreate();
        dataset = spark.read().option("header", "true").csv("NYC_SAMPLE_DATA.csv");
//        dataset.show();
        buildMapper("Vehicle Make", spark);
        buildMapper("Vehicle Body Type", spark);
        buildMapper("Issuing Agency", spark);
        buildMapper("Vehicle Expiration Date", spark);
    }

    public static void buildMapper(String columnName, SparkSession spark) {
        Dataset<Row> uniqueValues = dataset.select(columnName).distinct().sort().withColumn("id", monotonically_increasing_id());
        uniqueValues.write().csv(String.format("val_%s", columnName.toLowerCase().replace(" ", "_")));
    }

    //function for going through column and creating numerical equivalents to string values
}
