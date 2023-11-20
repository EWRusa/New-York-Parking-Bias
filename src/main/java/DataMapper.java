import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;
import scala.collection.immutable.HashMap;
import scala.collection.immutable.Map;

import java.util.Iterator;

public final class DataMapper {
    static String[] featuresToCapture = {"Vehicle Make", "Vehicle Body Type",
            "Issuing Agency","Vehicle Expiration Date", "Plate Type", "Street Name", "Intersecting Street"};

    public static void main(String[] args) {
        String predictedLabel = "Vehicle Make";

        String[] featuresToFix = {"Vehicle Make", "Vehicle Body Type",
                "Issuing Agency","Vehicle Expiration Date", "Plate Type", "Street Name", "Intersecting Street"};

        SparkSession spark = SparkSession
                .builder()
                .appName("DataMapper").master("local")
                .getOrCreate();
        Dataset<Row> dataset = spark.read().option("header", "true")
                .csv("NYC_SAMPLE_DATA.csv").persist(StorageLevel.MEMORY_AND_DISK());

        //UNTESTED but should work to remap values on a dataset
        for (String feature: featuresToFix) {
            dataset.withColumn(feature, dataset.col(feature).apply(classifierToRDD(feature, spark)));
        }

        JavaRDD<LabeledPoint> dataForRandomForest = dataset.toJavaRDD()
                .map(row -> new LabeledPoint(row.getDouble(row.fieldIndex(predictedLabel)), vectorBuilder(row)));

        dataForRandomForest.saveAsTextFile(String.format("random_forest_dataset_%s",predictedLabel.toLowerCase().replace(" ", "_")));
        spark.stop();
    }

    public static DenseVector vectorBuilder(Row row) {
        double[] vals = new double[featuresToCapture.length];

        for (int i = 0; i < vals.length; i++) vals[i] = row.getDouble(row.fieldIndex(featuresToCapture[i]));

        return new DenseVector(vals);
    }

    public static Map<String, Double> classifierToRDD(String columnName, SparkSession spark) {
        Dataset<Row> column = spark.read().option("header", "false")
                .csv(String.format("val_%s", columnName.toLowerCase().replace(" ", "_")));

        HashMap<String, Double> map = new HashMap<>();

        Iterator<Row> itr = column.toLocalIterator();
        while (itr.hasNext()) {
            Row r = itr.next();
            map.$plus(new Tuple2<String, Double>(r.getString(0), r.getDouble(1)));
        }

        return map;
    }
}
