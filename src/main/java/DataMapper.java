import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import scala.collection.immutable.HashMap;
import scala.collection.immutable.Map;

import java.util.Iterator;

public final class DataMapper {

    public void columnFixer() {
        String[] capturedFeatures = {"Vehicle Make", "Vehicle Body Type",
                "Issuing Agency","Vehicle Expiration Date", "Plate Type", "Street Name", "Intersecting Street"};

        String[] featuresToFix = {"Vehicle Make", "Vehicle Body Type",
                "Issuing Agency","Vehicle Expiration Date", "Plate Type", "Street Name", "Intersecting Street"};

        SparkSession spark = SparkSession
                .builder()
                .appName("ClassificationMapper").master("local")
                .getOrCreate();
        Dataset<Row> dataset = spark.read().option("header", "true").csv("NYC_SAMPLE_DATA.csv");

        //UNTESTED but should work to remap values on a dataset
        dataset.withColumn("Vehicle Make", dataset.col("Vehicle Make").apply(classifierToRDD("Vehicle Make", spark)));


    }

    public Map<String, Double> classifierToRDD(String columnName, SparkSession spark) {
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
