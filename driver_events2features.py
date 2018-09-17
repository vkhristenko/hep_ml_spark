import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import logging

import utils

if __name__ == "__main__":
    spark = SparkSession\
            .builder\
            .appName("ml pipeline: feature engineering")\
            .getOrCreate()

    # config logging
    logging.basicConfig(level = logging.INFO, format="[PYDRIVER] %(message)s")

    # we must have 4 args
    if len(sys.argv) != 5:
        print sys.argv
        sys.exit("incorrect number of cli arguments")

    # get the path to data @eos
    path_qcd = sys.argv[1]
    path_ttbar = sys.argv[2]
    path_wjets = sys.argv[3]
    path_output = sys.argv[4]

    logging.info("path for qcd dataset: %s" % path_qcd)
    logging.info("path for ttbar dataset: %s" % path_ttbar)
    logging.info("path for wjets dataset: %s" % path_wjets)
    logging.info("path to output dataset: %s" % path_output)

    # read in the data
    sparkroot = "org.dianahep.sparkroot.experimental"
    events_qcd = spark.read.format(sparkroot).load(path_qcd)
    events_ttbar = spark.read.format(sparkroot).load(path_ttbar)
    events_wjets = spark.read.format(sparkroot).load(path_wjets)

    logging.info("read in all the features' datasets")

    # we need to select only these columns
    requiredColumns = ["EFlowTrack", "MuonTight_size", "Electron_size", 
                       "EFlowNeutralHadron", "EFlowPhoton", "Electron", 
                       "MuonTight", "MissingET", "Jet"]

    # add a label and combine the data frames into 1
    label_qcd = 0
    label_ttbar = 1
    label_wjets = 2
    tmp_qcd = events_qcd.select(requiredColumns)\
        .toDF(*requiredColumns).withColumn("label", lit(label_qcd))
    tmp_ttbar = events_ttbar.select(requiredColumns)\
        .toDF(*requiredColumns).withColumn("label", lit(label_ttbar))
    tmp_wjets = events_wjets.select(requiredColumns)\
        .toDF(*requiredColumns).withColumn("label", lit(label_wjets))
    events = tmp_qcd.union(tmp_ttbar).union(tmp_wjets)

#    print "number of events = {nevents}".format(nevents = events.count())

    logging.info("performed the union of all the input datasets")

    # pipeline itself
    features = events\
            .rdd\
            .map(utils.convert)\
            .filter(lambda row: len(row) > 0)\
            .toDF()

    logging.info("built the pipeline and converted back into the data frame")

    # write to disk: note this will trigger the pipeline as a whole!
    features\
            .write\
            .parquet(path_output, mode="overwrite")

    logging.info("all done!")
