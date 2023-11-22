SPARK_MASTER=jackson:30315

boot:
	start-dfs.sh
	start-yarn.sh
	start-master.sh
	start-workers.sh

compile:
	mvn package

#usage : make run SPARK_MASTER=jackson:30315
# or whatever your spark master is
run-classification: compile
	spark-submit --class ClassificationMapper --master spark://$(SPARK_MASTER) target/ParkingRandomForest-1.0-SNAPSHOT.jar
run-mapper: compile
	spark-submit --class DataMapper --master spark://$(SPARK_MASTER) target/ParkingRandomForest-1.0-SNAPSHOT.jar

unboot:
	stop-yarn.sh
	stop-dfs.sh
	stop-master.sh
	stop-workers.sh