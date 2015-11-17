---
title: "README"
author: "RDSN"
date: "17 October 2015"
output: html_document
---

&nbsp;
&nbsp;
&nbsp;

## HOW TO MAKE IT WORK

Through the code written in the R file, here the work performed :

 1. First, unzip the data from  https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip 
in your working directory. A folder called "UCI HAR Dataset" will be created in your working directory. Make sure the folder "UCI HAR Dataset" and the run_analysis.R script are both in the current working directory.
 2. Second, use source("run_analysis.R") command in RStudio.
 3. Third, you will find one output file is generated in the current working directory:
    * tidyDataSet.txt (225 KB): it contains a data frame with 180 rows and 68 columns.
 4. Finally, use 
data <- read.table("tidyDataSet.txt", header = TRUE, stringsAsFactors = FALSE) 
command in RStudio to read the file. Since we are required to get the average of each variable for each activity and each subject, and there are 6 activities in total and 30 subjects in total, we have 180 rows with all combinations for each of the 66 features. Note that there are 68 columns, the 2 first corresponding to the activity label and the ID of the subject (66 + 2 = 68 columns).

Note :
once you have sourced "run_analysis.R" into RStudio, you have direct access to 2 dataframes, without having to read any other file :

 * a dataframe called "mat" whcich contains the dataframe available in the tidyDataSet.txt file
 * a global dataframe called "mergeDataFinal" that contains all the variables (68) without any average calculated for each activity and each subject. This dataset is a 10299 * 68 dataset.

&nbsp;

## HOW DOES THE "run_analysis.R" FILE WORKS

### Step1 - Merger of the training and the test sets to create one data set.

 * First we read the different txt files and we get sure that the dimensions corresponds to what we expect:
    * X_train.txt : dim = 7352 * 561
    * subject_train.txt : dim = 7352 * 1
    * y_train.txt : dim = 7352 * 1
    * X_test.txt : dim = 2947 * 561
    * subject_test.txt : dim = 2947 * 1
    * y_test.tx : dim = 2947 * 1

 * Then We merge the training and the test datasets
(We have now merged the training and the test. We still have to complete the dataset by adding the Subject column and the Label column. We'll do this after the step 2 so that it's easier to remove all the non-mean and non-standard measuring columns.)

### Step2 - Extraction of only the measurements on the mean and standard deviation for each measurement.

 * We first read the file containing the features names
 * Then we get the indices for all the features with mean and std measurements
 * We make the new dataset by selecting only the features we're interested in and that we have selected just before
 * Then we add the Label and the Subject columns to the dataset. But before, we are going to put features names on top of the columns of the set.

### Step 3 - Use of descriptive activity names to name the activities in the data set

 * We first read the activity labels dataset
 * Then we substitute the "-" by "" in the labels
 * Finally, for the 2 labels which are composed of 2 words, we change the first letter of the second word by an uppercase

 * We then add the activity column by merging
 * And we add the activity column to the dataset by merging
 * We delete the first column of the dataset which contains the number associated with the activity, which we don't need because we've add the activity name.

### Step 4 - Appropriate labeling of the data set with descriptive variable names.

For each name in the features list :
 
 * We delete parentheses
 * Wer replace "mean" by "Mean"
 * We replace "std" by "Std"
 * We delete the "-"
 * We then reorder the data columns

The dataset created is named **mergeDataFinal**

### Step 5 - From the data set in step 4, creation of a second, independent tidy data set with the average of each variable for each activity and each subject.

 * Here, we create a matrix to present the data.
As we compute the average for each activity and each subject, the matrix must
have 180 rows = 6 * 30 (6 activities and 30 subjects). The number of columns is equal to the number of features. 
This matrix is named **mat**
 * Finally, we write this "mat" dataframe into the "tidyDataSet.txt" file which is created in our working directory.
 
&nbsp;

## FILES AVAILABE IN THIS REPO

 * **Readme.md** : This file explain the whole context and how the different files work and are linked together.
 * **CodeBook.md** : This file describes the variables, the data, and the process performed in the R file so that to clean the data up.
 * **run_analysis.R** : this file contains the code written in R for performing the cleaning up of the data
 * **tidyDataSet.txt** : this file contains an independent tidy data set with the average of each variable for each activity and each subject. This data set has been performed through the 5th step coded at the end of the run_analysis.R file.
