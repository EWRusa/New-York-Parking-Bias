import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

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

        //I believe these are the two functions you are talking about in the old version of the code. I could not figure out a lot but hope to soon
        //JavaPairRDD<Double, Double> predictionAndLabel = dataFor2023.mapToPair((LabeledPoint labeledPoint) -> new Tuple2(modelToTest.predict(labeledPoint.features()), labeledPoint.label()));

        //double testErr = predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / (double) dataFor2023.count();



        double accuracy = 0.0; //placeholder
        Logger logger = Logger.getRootLogger();

        logger.info(String.format("Error for Predicting 2023 %s: %.6f", datapathLabel, 1.0 - accuracy));

        spark.stop();
    }
}
