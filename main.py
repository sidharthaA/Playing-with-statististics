
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from numpy.random import choice
from numpy import mean
import numpy as np
import scipy.stats


def make_confidence_interval(sample):
    confidence_level = 0.90
    degrees_freedom = len(sample) - 1
    sample_mean = np.mean(sample)
    sample_standard_error = scipy.stats.sem(sample)
    confidence_interval = scipy.stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
    return confidence_interval


def make_confidence_interval_median(sample):
    confidence_level = 0.90
    degrees_freedom = len(sample) - 1
    sample_median = np.median(sample)
    sample_standard_error = scipy.stats.sem(sample)
    confidence_interval = scipy.stats.t.interval(confidence_level, degrees_freedom, sample_median,
                                                 sample_standard_error)
    return confidence_interval


# read the values from file
df = pd.read_excel(r'Real estate valuation data set.xlsx')
column_list = df.columns


# function to calculate the mean
def get_mean(df, column):
    # mean_table=[]
    return df[column].mean()


# function to calculate the median
def get_median(df, column):
    # mean_table=[]
    return df[column].median()


# mean of dataset
# Taking samples and denoting each to a different dataframe
meanTable = []
medianTable = []
mean10 = []
median10 = []
df1 = df.sample(n=10)
df2 = df.sample(n=100)
df3 = df.sample(n=200)
for col in column_list:
    meanTable.append(get_mean(df, col))
    medianTable.append(get_median(df, col))
    mean10.append(get_mean(df1, col))
    median10.append(get_median(df1, col))

print("mean of the dataset")
print(meanTable)

print("median of the dataset")
print(medianTable)
# mean10
print('mean10')
print(mean10)

# median10
print("median10")
print(median10)


# function to convert to numpy
def convert_to_numpy(df, col):
    return df[col].to_numpy()


# Dictionaries to store confidence interval of each sampled dataframe
convert_rows1 = {}
convert_rows2 = {}
convert_rows3 = {}
convert_rows = {}
# Storing each numpy array in the dictionary
for col in column_list:
    convert_rows1[col] = convert_to_numpy(df1, col)
    convert_rows2[col] = convert_to_numpy(df2, col)
    convert_rows3[col] = convert_to_numpy(df3, col)
    convert_rows[col] = convert_to_numpy(df, col)

# Confidence interval to store each method is used to compute for each row
confidence_interval = {}
confidence_interval_interval_median = {}
for i in convert_rows1.keys():
    confidence_interval[i] = make_confidence_interval(convert_rows1[i])
    confidence_interval_interval_median[i] = make_confidence_interval_median(convert_rows1[i])

# Function to print output for each dictionary
def print_output(string_val, dictionary):
    print(string_val)
    for i in dictionary.keys():
        print(dictionary[i])

print_output('mean10int', confidence_interval)
print_output('median10int', confidence_interval_interval_median)
confidence_interval100 = {}
confidence_interval_interval_median100 = {}
for i in convert_rows2.keys():
    confidence_interval100[i] = make_confidence_interval(convert_rows2[i])
    confidence_interval_interval_median100[i] = make_confidence_interval_median(convert_rows2[i])
print_output('mean100int', confidence_interval100)
print_output('median100int', confidence_interval_interval_median100)
confidence_interval200 = {}
confidence_interval_interval_median200 = {}
for i in convert_rows3.keys():
    confidence_interval200[i] = make_confidence_interval(convert_rows3[i])
    confidence_interval_interval_median200[i] = make_confidence_interval_median(convert_rows3[i])
print_output('mean200int', confidence_interval200)
print_output('median200int', confidence_interval_interval_median200)

# bootstrapping for mean and median for all 3 varieties of samplings
X1mean10 = []
X2mean10 = []
X3mean10 = []
X4mean10 = []
X5mean10 = []
X6mean10 = []
Ymean10 = []

X1mean100 = []
X2mean100 = []
X3mean100 = []
X4mean100 = []
X5mean100 = []
X6mean100 = []
Ymean100 = []

X1mean200 = []
X2mean200 = []
X3mean200 = []
X4mean200 = []
X5mean200 = []
X6mean200 = []
Ymean200 = []

X1median10 = []
X2median10 = []
X3median10 = []
X4median10 = []
X5median10 = []
X6median10 = []
Ymedian10 = []

X1median100 = []
X2median100 = []
X3median100 = []
X4median100 = []
X5median100 = []
X6median100 = []
Ymedian100 = []

X1median200 = []
X2median200 = []
X3median200 = []
X4median200 = []
X5median200 = []
X6median200 = []
Ymedian200 = []

# Iterating over each item sample to compute mean
for item in range(1000):
    dfi = df1.sample(n=10,replace=True)
    X1mean10.append(dfi['X1 transaction date'].mean())
    X2mean10.append(dfi['X2 house age'].mean())
    X3mean10.append(dfi['X3 distance to the nearest MRT station'].mean())
    X4mean10.append(dfi['X4 number of convenience stores'].mean())
    X5mean10.append(dfi['X5 latitude'].mean())
    X6mean10.append(dfi['X6 longitude'].mean())
    Ymean10.append(dfi['Y house price of unit area'].mean())

    dfiMed = df1.sample(n=10,replace=True)
    X1median10.append(dfiMed['X1 transaction date'].median())
    X2median10.append(dfiMed['X2 house age'].median())
    X3median10.append(dfiMed['X3 distance to the nearest MRT station'].median())
    X4median10.append(dfiMed['X4 number of convenience stores'].median())
    X5median10.append(dfiMed['X5 latitude'].median())
    X6median10.append(dfiMed['X6 longitude'].median())
    Ymedian10.append(dfiMed['Y house price of unit area'].median())

    df100 = df2.sample(n=100,replace=True)
    X1mean100.append(df100['X1 transaction date'].mean())
    X2mean100.append(df100['X2 house age'].mean())
    X3mean100.append(df100['X3 distance to the nearest MRT station'].mean())
    X4mean100.append(df100['X4 number of convenience stores'].mean())
    X5mean100.append(df100['X5 latitude'].mean())
    X6mean100.append(df100['X6 longitude'].mean())
    Ymean100.append(df100['Y house price of unit area'].mean())

    df100Med = df2.sample(n=100,replace=True)
    X1median100.append(df100Med['X1 transaction date'].median())
    X2median100.append(df100Med['X2 house age'].median())
    X3median100.append(df100Med['X3 distance to the nearest MRT station'].median())
    X4median100.append(df100Med['X4 number of convenience stores'].median())
    X5median100.append(df100Med['X5 latitude'].median())
    X6median100.append(df100Med['X6 longitude'].median())
    Ymedian100.append(df100Med['Y house price of unit area'].median())

    df200 = df3.sample(n=200,replace=True)
    X1mean200.append(df200['X1 transaction date'].mean())
    X2mean200.append(df200['X2 house age'].mean())
    X3mean200.append(df200['X3 distance to the nearest MRT station'].mean())
    X4mean200.append(df200['X4 number of convenience stores'].mean())
    X5mean200.append(df200['X5 latitude'].mean())
    X6mean200.append(df200['X6 longitude'].mean())
    Ymean200.append(df200['Y house price of unit area'].mean())

    df200Med = df3.sample(n=200,replace=True)
    X1median200.append(df200Med['X1 transaction date'].median())
    X2median200.append(df200Med['X2 house age'].median())
    X3median200.append(df200Med['X3 distance to the nearest MRT station'].median())
    X4median200.append(df200Med['X4 number of convenience stores'].median())
    X5median200.append(df200Med['X5 latitude'].median())
    X6median200.append(df200Med['X6 longitude'].median())
    Ymedian200.append(df200Med['Y house price of unit area'].median())


# Function to print
def print_bootstrap(str_val, arraylist):
    print(str_val)
    for arr in arraylist:
        arr.sort()
        print(arr[49], arr[949])


print_bootstrap("mean10boot", (X1mean10, X2mean10, X3mean10, X4mean10, X5mean10, X6mean10, Ymean10))
print_bootstrap("med10boot", (X1median10, X2median10, X3median10, X4median10, X5median10, X6median10, Ymedian10))
print_bootstrap("mean100boot", (X1mean100, X2mean100, X3mean100, X4mean100, X5mean100, X6mean100, Ymean100))
print_bootstrap("med100boot",
                (X1median100, X2median100, X3median100, X4median100, X5median100, X6median100, Ymedian100))
print_bootstrap("mean200boot", (X1mean200, X2mean200, X3mean200, X4mean200, X5mean200, X6mean200, Ymean200))
print_bootstrap("med200boot",
                (X1median200, X2median200, X3median200, X4median200, X5median200, X6median200, Ymedian200))

data = np.array([convert_rows[column_list[0]], convert_rows[column_list[1]], convert_rows[column_list[2]]
                    , convert_rows[column_list[3]], convert_rows[column_list[4]],
                 convert_rows[column_list[5]], convert_rows[column_list[6]]])
covMatrix = np.cov(data, bias=True)
print(covMatrix)


# histograms
def construct_hist(col: str, df_array):
    for i in df_array:
        plt.figure()
        plt.hist(i[col])
        if len(i) == 10:
            plt.savefig('hist10_' + "_".join(col.split()) + '.svg')
        elif len(i) == 100:
            plt.savefig('hist100_' + "_".join(col.split()) + '.svg')
        elif len(i) == 200:
            plt.savefig('hist200_' + "_".join(col.split()) + '.svg')
        else:
            plt.savefig('hist_' + "_".join(col.split()) + '.svg')
    plt.figure()
    plt.hist([df_array[0][col], df_array[1][col], df_array[2][col], df_array[3][col]], bins=30, stacked=True,
             density=True)
    plt.savefig('Stacked_hist.svg')
    plt.figure(figsize=[15, 8])
    plt.hist([df_array[0][col], df_array[1][col], df_array[2][col], df_array[3][col]], bins=30,
             density=True)
    plt.savefig('Side_hist.svg')


construct_hist('X5 latitude', [df, df1, df2, df3])