# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colrs

from pyspark.sql import SparkSession
import pyspark.sql.functions as spark_functions
import pyspark.sql.types as spark_types
from pyspark.sql.functions import col, pandas_udf, udf

# %%
USER = os.getlogin()
WORKING_DIR = f'/home/{USER}/data/Land_use'
DATA_DIR = f'{WORKING_DIR}/data'
METROPOLES_SHAPE = f'{DATA_DIR}/cities'
IMG_DIR = f'{WORKING_DIR}/images'

# %%
spark = SparkSession.builder\
    .master('spark://santiago:7077')\
    .appName('Land use - SRCA iris median week')\
    .config('spark.network.timeout', 300)\
    .config('spark.dynamicAllocation.enabled', 'true')\
    .config('spark.shuffle.service.enabled', 'true')\
    .config('spark.dynamicAllocation.initialExecutors', 1)\
    .config('spark.dynamicAllocation.maxExecutors', 20)\
    .config('spark.dynamicAllocation.minExecutors', 0)\
    .config('spark.driver.maxResultSize', '120g')\
    .config('spark.executor.cores', 1)\
    .config('spark.executor.memory', '3g')\
    .config('spark.memory.fraction', 0.6)\
    .config('spark.cores.max', 20)\
    .config('spark.executor.memoryOverhead', '8g')\
    .config('spark.driver.memoryOverhead', '8g')\
    .getOrCreate()

spark.conf.set('spark.sql.session.timeZone', 'Europe/Paris')

# %%
CITY_NAME = 'Paris'
df_mask = pd.read_pickle(f'{DATA_DIR}/df_masks.pkl')
city_row = df_mask[df_mask['name'] == CITY_NAME].iloc[0]

city_shape = city_row['shape']
city_mask = city_row['mask']
city_left_x = city_row['left_x']
city_bottom_y = city_row['bottom_y']

# %%
filename = f'hdfs://santiago:9000/land_use/{CITY_NAME}_iris_traffic_maps_median_week.parquet'
sdf_traffic = spark.read.parquet(filename)
sdf_traffic = sdf_traffic.drop('traffic_map')
#sdf_traffic = sdf_traffic.withColumnRenamed('median_week_traffic_map', 'traffic_map')
sdf_traffic.show(2)

# %%
# some_rows = sdf_traffic.take(10)
# data = []
# for row in some_rows:
#     data.append(row.asDict())

# df = pd.DataFrame(data)
# sdf_small = spark.createDataFrame(df)
# sdf_small.show(2)

# %% [markdown]
# ## Symmetric RCA

# %%
schema_traffic_map = spark_types.ArrayType(spark_types.FloatType())

@pandas_udf(schema_traffic_map)
def total_traffic_map(traffic_maps_iris: pd.Series)-> schema_traffic_map:
    traffic_maps_iris = traffic_maps_iris.apply(lambda traffic_map: np.array(list(traffic_map)))
    traffic_map = traffic_maps_iris.sum(axis=0)

    return traffic_map.tolist()

# %%
schema_sum = spark_types.FloatType()

@udf(schema_sum)
def apply_sum(traffic_map) -> schema_sum:
    traffic_map = np.array(list(traffic_map))
    return float(np.sum(traffic_map))

# %%
sdf_traffic_time = sdf_traffic.groupBy('city', 'time', 'day_of_week').agg(total_traffic_map('traffic_map_iris').alias('total_traffic_map_iris'))
sdf_traffic_time = sdf_traffic_time.withColumn('total_traffic', apply_sum('total_traffic_map_iris'))
sdf_traffic_time.show(2)

# %%
df_traffic_time = sdf_traffic_time.toPandas()
df_traffic_time['total_traffic_map_iris'] = df_traffic_time['total_traffic_map_iris'].apply(lambda traffic_map: np.array(traffic_map))
df_traffic_time.head(2)

# %%
# Tij = traffic of app i in location j
# Tj = traffic of all apps in location j
# Ti = traffic of app i in all locations
# T = traffic of all apps in all locations

schema_traffic_map = spark_types.ArrayType(spark_types.DoubleType())

@udf(returnType=schema_traffic_map)
def compute_SRCA(time, day, traffic_map) -> schema_traffic_map:

    traffic_map = np.array(list(traffic_map))
    
    Tij = traffic_map
    Tj = df_traffic_time[(df_traffic_time['time'] == time)&(df_traffic_time['day_of_week'] == day)].iloc[0]['total_traffic_map_iris']
    Ti = traffic_map.sum()
    T = Tj.sum()

    RCA = (Tij / Tj) / (Ti / T)
    SRCA = (RCA - 1) / (RCA + 1)
    #SRCA[ city_mask == 0 ] = 0
    return SRCA.tolist()

# %%
print('*********** Computing SRCA ***********')
sdf_traffic = sdf_traffic.withColumn('iris_traffic_map_srca', compute_SRCA('time', 'day_of_week', 'traffic_map_iris'))
#sdf_traffic.show(2)
print('*********** writing ***********')
sdf_traffic.write.parquet(f'hdfs://santiago:9000/land_use/{CITY_NAME}_iris_srca_traffic_maps_median_week.parquet', mode='overwrite')
print('************  Done  **************')

# %%



