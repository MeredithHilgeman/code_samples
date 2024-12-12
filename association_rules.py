# Example Code to Start Looking at Association Rules

## Import libraries
import pandas as pd 
import numpy as np 
from mlxtend.frequent_patterns import apriori, association_rules

## Pull data into script
data = pd.read_csv("data.csv")

## Pre-processing would happen here

# - check for nulls
# - check for duplicates
# - check data types / get summary of data
# - each record is an ID, multiple binary columns, market basket for example

## Association Rules
ap.apriori(data, min_support = 0.01, use_colnames = True, verbose = 1)

association_rules(ap, metric = "lift", min_threshold = 0.01)