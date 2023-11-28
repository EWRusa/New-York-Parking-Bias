import org.apache.log4j.Logger;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

public class RandomForestTester {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("RandomForestTester").master("yarn")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        String datapathLabel = args[0];
        //pulls specifically made 2023 data
        JavaRDD<LabeledPoint> dataFor2023 = MLUtils.loadLabeledPoints(jsc.sc(), String.format("random_forest_dataset_%s_2023",datapathLabel.toLowerCase().replace(" ", "_"))).toJavaRDD();

        RandomForestModel modelToTest = RandomForestModel.load(jsc.sc(), String.format("random_forest_model_%s", datapathLabel.toLowerCase().replace(" ", "_")));

        JavaPairRDD<Double, Double> predictionAndLabel =
                dataFor2023.mapToPair((LabeledPoint labeledPoint) -> new Tuple2(modelToTest.predict(labeledPoint.features()), labeledPoint.label()));
        double testErr =
                predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / (double) dataFor2023.count();

        Logger logger = Logger.getRootLogger();

        logger.info(String.format("Error for Predicting 2023 %s: %.6f", datapathLabel, testErr));

        spark.stop();
    }
}
