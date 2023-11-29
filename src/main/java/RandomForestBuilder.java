import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassificationSummary;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorIndexer;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.io.IOException;

public class RandomForestBuilder {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .appName("RandomForestMaker").master("yarn")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
        String datapathLabel = args[0];

        Dataset<Row> data = spark.read().format("libsvm").load(String.format("random_forest_dataset_%s",datapathLabel.toLowerCase().replace(" ", "_")));

        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(data);
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(data);

        int numClasses = (int) spark.read().option("header", "false")
                .csv(String.format("val_%s", datapathLabel.toLowerCase().replace(" ", "_"))).count();

//        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        int numTrees = 60;
        String featureSubsetStrategy = "auto";
        String impurity = "gini";
        int maxDepth = 15;
        int maxBins = numClasses * 2;

        //this is all untested currently

        //K-Folds
        int numSplits = 5;
        Dataset<Row>[] splits = data.randomSplit(kSplits(numSplits));

        Tuple2<RandomForestClassificationModel, Double>[] modelList = new Tuple2[numSplits];

        //K-Folds cross validator
        int seed = 12345;
        for (int i = 0; i < numSplits; i++) {

            Dataset<Row> testData = splits[i];
            Dataset<Row> trainingData = trainingDataBuilder(i, numSplits, splits);

            //this supposedly takes a long time to complete
            RandomForestClassifier classifier =  new RandomForestClassifier().setLabelCol("indexedLabel")
                    .setFeaturesCol("indexedFeatures").setMaxBins(maxBins).setNumTrees(numTrees).setMaxDepth(maxDepth)
                    .setImpurity(impurity).setFeatureSubsetStrategy(featureSubsetStrategy);

            RandomForestClassificationModel model = classifier.fit(trainingData);

            RandomForestClassificationSummary summary = model.evaluate(testData);
            modelList[i] = new Tuple2<>(model, summary.accuracy());
            System.out.println("FINISHED " + (i+1) + " RANDOM FOREST MODEL");
        }

        //finds model with the smallest error
        double currentMaxAccuracy = Double.MIN_VALUE;
        int currentMaxIndex = 0;
        for (int i = 0; i < modelList.length; i++) {
            if (modelList[i]._2() > currentMaxAccuracy) {
                currentMaxAccuracy = modelList[i]._2();
                currentMaxIndex = i;
                System.out.println("NEW MAX ACCURACY FOUND " + modelList[i]._2());
            }
        }

        //saves the best model
        try {
            modelList[currentMaxIndex]._1().save(String.format("random_forest_model_%s",(datapathLabel).toLowerCase().replace(" ", "_")));
        } catch (IOException e) {
            System.out.println("CRITICAL ERROR UNABLE TO SAVE MODEL");
            throw new RuntimeException(e);
        }
    }

    public static Dataset<Row> trainingDataBuilder(int testingDataParam, int numSplits, Dataset<Row>[] splits) {
        Dataset<Row> trainingData = splits[(testingDataParam + 1) % numSplits];

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
