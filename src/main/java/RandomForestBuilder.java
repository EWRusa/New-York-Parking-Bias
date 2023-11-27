import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

public class RandomForestBuilder {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("RandomForestMaker").master("yarn")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        String datapathLabel = args[0];

        JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), String.format("random_forest_dataset_%s",datapathLabel.toLowerCase().replace(" ", "_"))).toJavaRDD();

        int numClasses = (int) spark.read().option("header", "false")
                .csv(String.format("val_%s", datapathLabel.toLowerCase().replace(" ", "_"))).count();
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        int numTrees = 60;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        int maxDepth = 15;
        int maxBins = 1000;

        //this is all untested currently

        //K-Folds
        int numSplits = 5;
        JavaRDD<LabeledPoint>[] splits = data.randomSplit(kSplits(numSplits));

        Tuple2<RandomForestModel, Double>[] modelList = new Tuple2[numSplits];

        //K-Folds cross validator
        int seed = 12345;
        for (int i = 0; i < numSplits; i++) {

            JavaRDD<LabeledPoint> testData = splits[i];
            JavaRDD<LabeledPoint> trainingData = trainingDataBuilder(i, numSplits, splits);

            //this supposedly takes a long time to complete
            RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses,
                    categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);


            JavaPairRDD<Double, Double> predictionAndLabel =
                    testData.mapToPair((LabeledPoint labeledPoint) -> new Tuple2(model.predict(labeledPoint.features()), labeledPoint.label()));
            double testErr =
                    predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / (double) testData.count();
            //assigns model and error to a spot in the model list, this will be used later to find the lowest error;
            modelList[i] = new Tuple2<>(model, testErr);
        }

        //finds model with the smallest error
        double currentMinError = Double.MAX_VALUE;
        int currentMinIndex = 0;
        for (int i = 0; i < modelList.length; i++) {
            if (modelList[i]._2() < currentMinError) {
                currentMinError = modelList[i]._2();
                currentMinIndex = i;
            }
        }

        //saves the best model
        modelList[currentMinIndex]._1().save(jsc.sc(), String.format("random_forest_model_%s",datapathLabel.toLowerCase().replace(" ", "_")));
    }

    public static JavaRDD<LabeledPoint> trainingDataBuilder(int testingDataParam, int numSplits, JavaRDD<LabeledPoint>[] splits) {
        JavaRDD<LabeledPoint> trainingData = splits[(testingDataParam + 1) % numSplits];

        for (int i = 0; i < numSplits; i++) {
            if (i != testingDataParam || i != (testingDataParam + 1) % numSplits) {
                trainingData = trainingData.union(splits[i]);
            }
        }

        return trainingData;

    }

    public static double[] kSplits(int numSplits) {
        double portionData = 1.0 / numSplits;
        double[] dataSplit = new double[numSplits];
        for (int i = 0; i < numSplits; i++) {
            dataSplit[i] = portionData;
        }
        return dataSplit;
    }
}
