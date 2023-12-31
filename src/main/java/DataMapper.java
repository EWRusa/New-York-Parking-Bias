import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.rdd.RDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.expressions.UserDefinedFunction;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.storage.StorageLevel;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.udf;

public final class DataMapper {


    public static void main(String[] args) {
        String predictedLabel = args[0].replace("_"," ");

        boolean is2023 = args[1].contains("true");
        String marker2023 = "";
        if (is2023) marker2023 = " 2023";

        String[] featuresToFix = {"Vehicle Make", "Vehicle Body Type",
                "Issuing Agency","Vehicle Expiration Date", "Plate Type", "Street Name", "Intersecting Street", "Vehicle Color", "Vehicle Year", "Violation Description"};

        String[] allUsedFeatures = {"Vehicle Make", "Vehicle Body Type",
                "Issuing Agency","Vehicle Expiration Date", "Plate Type", "Street Name", "Intersecting Street", "Vehicle Color", "Vehicle Year", "Violation Description"};

        ArrayList<String> captureList = new ArrayList<>();

        for (int i = 0; i < allUsedFeatures.length; i++) {
            if (!allUsedFeatures[i].equals(predictedLabel))
                captureList.add(allUsedFeatures[i]);
        }

        String[] featuresToCapture = captureList.toArray(new String[captureList.size()]);

        String[] datapaths = {"/input/Parking_Violations_Issued_-_Fiscal_Year_2023_20231111.csv","/input/Parking_Violations_Issued_-_Fiscal_Year_2022_20231111.csv", "/input/Parking_Violations_Issued_-_Fiscal_Year_2021_20231111.csv"};

        //need to change to read from multiple data years
        SparkSession spark = SparkSession
                .builder()
                .appName("DataMapper").master("yarn")
                .getOrCreate();
        Dataset<Row> dataset;
        if (is2023) dataset = spark.read().option("header", "true")
                .csv(datapaths[0]).persist(StorageLevel.MEMORY_AND_DISK());
        else dataset = spark.read().option("header", "true")
                .csv(datapaths[1]).union(spark.read().option("header", "true")
                        .csv(datapaths[2])).persist(StorageLevel.MEMORY_AND_DISK());

        //UNTESTED but should work to remap values on a dataset
        for (String feature: featuresToFix) {
            HashMap<String, String> map = classifierToRDD(feature, spark);

            UserDefinedFunction replaceValuesUDF = udf((value) -> map.getOrDefault(value, "-1.0"), DataTypes.StringType);

            dataset = dataset.withColumn(feature, replaceValuesUDF.apply(col(feature)));
//            dataset.show(2);
        }

        RDD<LabeledPoint> dataForRandomForest = dataset.toJavaRDD()
                .map(row -> new LabeledPoint(Double.parseDouble(row.getString(row.fieldIndex(predictedLabel))), vectorBuilder(row, featuresToCapture))).rdd();

        MLUtils.saveAsLibSVMFile(dataForRandomForest,String.format("random_forest_dataset_%s",(predictedLabel + marker2023).toLowerCase().replace(" ", "_")));

//        dataForRandomForest.saveAsTextFile(String.format("random_forest_dataset_%s",(predictedLabel + marker2023).toLowerCase().replace(" ", "_")));
        spark.stop();
    }

    public static DenseVector vectorBuilder(Row row, String[] featuresToCapture) {
        double[] vals = new double[featuresToCapture.length];

        for (int i = 0; i < vals.length; i++) {
            try {
                vals[i] = Double.parseDouble(row.getString(row.fieldIndex(featuresToCapture[i])));
            } catch (Exception e) {
                vals[i] = -1.0;
            }
        }

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
