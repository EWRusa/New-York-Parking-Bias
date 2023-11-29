import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;

import static org.apache.spark.sql.functions.monotonically_increasing_id;

public final class ClassificationMapper {

    public static void main(String[] args) {
        String[] columnsToFix = {"Vehicle Make", "Vehicle Body Type",
                "Issuing Agency","Vehicle Expiration Date", "Plate Type", "Street Name", "Intersecting Street", "Vehicle Color", "Vehicle Year", "Violation Description"};

        //need to change to build off of all data years
        String[] datapaths = {"input/Parking_Violations_Issued_-_Fiscal_Year_2023_20231111.csv","input/Parking_Violations_Issued_-_Fiscal_Year_2022_20231111.csv", "input/Parking_Violations_Issued_-_Fiscal_Year_2021_20231111.csv"};

        SparkSession spark = SparkSession
                .builder()
                .appName("ClassificationMapper").master("yarn")
                .getOrCreate();
        Dataset<Row> dataset = spark.read().option("header", "true").csv(datapaths[2])
                .union(spark.read().option("header", "true").csv(datapaths[1]))
                .union(spark.read().option("header", "true").csv(datapaths[0]));
        dataset.persist(StorageLevel.MEMORY_AND_DISK());
//        dataset.show();
        for (String column: columnsToFix) {
            buildMapper(column, dataset);
        }

        dataset.unpersist();

        spark.stop();
    }

    public static void buildMapper(String columnName, Dataset<Row> dataset) {
        Dataset<Row> uniqueValues = dataset.select(columnName).distinct().sort().withColumn("id", monotonically_increasing_id());
        uniqueValues.write().csv(String.format("val_%s", columnName.toLowerCase().replace(" ", "_")));
    }

    //function for going through column and creating numerical equivalents to string values
}
