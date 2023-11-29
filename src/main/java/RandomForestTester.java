import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.concurrent.atomic.AtomicInteger;

public class RandomForestTester {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("RandomForestTester").master("local")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        String datapathLabel = args[0];
        //pulls specifically made 2023 data
        Dataset<Row> dataFor2023 = spark.read().format("libsvm").load(String.format("random_forest_dataset_%s_2023",datapathLabel.toLowerCase().replace(" ", "_")));

        RandomForestClassificationModel modelToTest = RandomForestClassificationModel.load(String.format("random_forest_model_%s",(datapathLabel).toLowerCase().replace(" ", "_")));
//        RandomForestModel modelToTest = RandomForestModel.load(jsc.sc(), String.format("random_forest_model_%s", datapathLabel.toLowerCase().replace(" ", "_")));

        Dataset<Row> predictions = modelToTest.transform(dataFor2023);

        //take this dataset and compare column of actual to predicted, pretty sure i already did this in an old version
        Dataset<Row> predictionsUnifiedToActual = dataFor2023.withColumn("predictedValue", predictions.apply(predictions.columns()[0]));

        predictionsUnifiedToActual.show(3);

        AtomicInteger countCorrect = new AtomicInteger();

        predictionsUnifiedToActual.foreach((Row row) -> rowCompare(row, countCorrect));

        double accuracy = (double) countCorrect.get() / (double) predictionsUnifiedToActual.count(); //placeholder
        Logger logger = Logger.getRootLogger();

        logger.info(String.format("Error for Predicting 2023 %s: %.6f", datapathLabel, 1.0 - accuracy));

        spark.stop();
    }

    private static void rowCompare(Row row, AtomicInteger countCorrect) {
        if (row.get(0).equals(row.get(2))) countCorrect.incrementAndGet();
    }
}
