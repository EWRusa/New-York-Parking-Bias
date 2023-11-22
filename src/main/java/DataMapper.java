import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;


import java.util.HashMap;
import java.util.Iterator;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.udf;

public final class DataMapper {
    static String[] featuresToCapture = { "Vehicle Body Type",
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
            HashMap<String, String> map = classifierToRDD(feature, spark);

            UserDefinedFunction replaceValuesUDF = udf((value) -> map.getOrDefault(value, "-1.0"), DataTypes.StringType);

            dataset.withColumn(feature, replaceValuesUDF.apply(col(feature)));
        }

        JavaRDD<LabeledPoint> dataForRandomForest = dataset.toJavaRDD()
                .map(row -> new LabeledPoint(row.getDouble(row.fieldIndex(predictedLabel)), vectorBuilder(row)));

        dataForRandomForest.saveAsTextFile(String.format("random_forest_dataset_%s",predictedLabel.toLowerCase().replace(" ", "_")));
        spark.stop();
    }

    public static DenseVector vectorBuilder(Row row) {
        double[] vals = new double[featuresToCapture.length];

        for (int i = 0; i < vals.length; i++) vals[i] = Double.parseDouble(row.getString(row.fieldIndex(featuresToCapture[i])));

        return new DenseVector(vals);
    }

    public static HashMap<String, String> classifierToRDD(String columnName, SparkSession spark) {
        Dataset<Row> column = spark.read().option("header", "false")
                .csv(String.format("val_%s", columnName.toLowerCase().replace(" ", "_")));

        HashMap<String, String> map = new HashMap<>();

        Iterator<Row> itr = column.toLocalIterator();
        while (itr.hasNext()) {
            Row r = itr.next();
            map.put(r.getString(0), r.getString(1));
        }

        return map;
    }
}