import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.param.Param;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.Iterator;
import java.util.concurrent.atomic.AtomicInteger;

public class RandomForestTester {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("RandomForestTester").master("yarn")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        String datapathLabel = args[0];
        //pulls specifically made 2023 data
        Dataset<Row> dataFor2023 = spark.read().format("libsvm").load(String.format("random_forest_dataset_%s_2023",datapathLabel.toLowerCase().replace(" ", "_")));

        RandomForestClassificationModel modelToTest = RandomForestClassificationModel.load(String.format("random_forest_model_%s",(datapathLabel).toLowerCase().replace(" ", "_"))).setFeaturesCol("features");
//        RandomForestModel modelToTest = RandomForestModel.load(jsc.sc(), String.format("random_forest_model_%s", datapathLabel.toLowerCase().replace(" ", "_")));

        Dataset<Row> predictions = modelToTest.transform(dataFor2023);

        

        long countCorrect = predictions.filter(predictions.col("label").equalTo(predictions.col("prediction"))).count();
        predictions.filter(predictions.col("label").equalTo(predictions.col("prediction"))).sort(predictions.col("prediction").desc()).show(20);
        double accuracyFor2023 = (double) countCorrect / (double) predictions.count(); //placeholder

//        System.out.println(String.format("Error for overall model %s: %.6f", datapathLabel, 1.0 - modelToTest.summary().accuracy()));
        System.out.println(String.format("Error for Predicting 2023 %s: %.6f", datapathLabel, 1.0 - accuracyFor2023));

        System.out.println(modelToTest.getProbabilityCol());

//        Param<String> probabilities = modelToTest.probabilityCol();
//        logger.info(probabilities.toString());
//
//        probabilities.

        spark.stop();
    }
}
