import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
import logging

import funcs_images as funcs

if __name__ == "__main__":
    spark = SparkSession\
            .builder\
            .appName("Abstract Image Build Up")\
            .getOrCreate()

    # config logging
    logging.basicConfig(level = logging.INFO, format="[PYDRIVER] %(message)s")

    # must have 4 args
    if len(sys.argv) != 5:
        sys.exit("Incorrect number of CLI arguments")

    # get the path to data in hdfs
    path_qcd = sys.argv[1]
    path_ttbar = sys.argv[2]
    path_wjets = sys.argv[3]
    path_output = sys.argv[4]
    
    logging.info("path qcd: %s" % path_qcd)
    logging.info("path ttbar: %s" % path_ttbar)
    logging.info("path wjets: %s" % path_wjets)

    # read in the data
    features_qcd = spark.read.format("parquet").load(path_qcd)
    features_ttbar = spark.read.format("parquet").load(path_ttbar)
    features_wjets = spark.read.format("parquet").load(path_wjets)

    logging.info("read in all the features' datasets")

    # add a label and combine the data frames into 1 
    label_qcd = 0
    label_ttbar = 1
    label_wjets = 2
    tmp_qcd = features_qcd.withColumn("label", lit(label_qcd))
    tmp_ttbar = features_ttbar.withColumn("label", lit(label_ttbar))
    tmp_wjets = features_wjets.withColumn("label", lit(label_wjets))
    features = tmp_qcd.union(tmp_ttbar).union(tmp_wjets)

    logging.info("performed the union of all the input datasets")

    # pipeline itself
    images = features\
            .rdd\
            .map(funcs.convert2image)\
            .toDF()

    logging.info("Built the pipeline and converted back into the data frame")

    # write to disk
    images\
            .write\
            .parquet(path_output, mode="overwrite")

    logging.info("All done!")
