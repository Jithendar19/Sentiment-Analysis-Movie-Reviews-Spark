# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 17:35:47 2017

@author: Jeethu
"""

import os, sys
os.environ['SPARK_HOME'] = "C:\Spark"
sys.path.append("C:\Spark\python")
sys.path.append("C:\Spark\python\lib")

from pyspark import SparkContext
import numpy as np
import collections
sc = SparkContext()

exampleRDD = sc.textFile("C:\Spark\README.md")

print exampleRDD.count()