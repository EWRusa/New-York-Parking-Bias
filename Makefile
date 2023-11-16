boot:
	start-dfs.sh
	start-yarn.sh
	start-master.sh
	start-workers.sh

compile:
	mvn package
#usage : make run SPARK_MASTER=jackson:30315
# or whatever your spark master is

unboot:
	stop-yarn.sh
	stop-dfs.sh
	stop-master.sh
	stop-workers.sh