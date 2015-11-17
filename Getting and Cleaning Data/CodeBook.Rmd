---
title: "CodeBook"
author: "RDSN"
date: "17 October 2015"
output: html_document
---

&nbsp;

## STUDY DESIGN

The data represent data collected from the accelerometers from the Samsung Galaxy S smartphone. A full description is available at the site where the data was obtained: 

<http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones>

The data used for this project are available here :
<https://d396qusza40orc.cloudfront.net/getdata%2Fprojectfiles%2FUCI%20HAR%20Dataset.zip>

Once you've unzip this document, you obtain a folder named "UCI HAR Dataset".
In this folder, you will find the following files :
 
 * 'README.txt'
 * 'features_info.txt': Shows information about the variables used on the feature vector.
 * 'features.txt': List of all features.
 * 'activity_labels.txt': Links the class labels with their activity name.
 * 'train/X_train.txt': Training set.
 * 'train/y_train.txt': Training labels.
 * 'test/X_test.txt': Test set.
 * 'test/y_test.txt': Test labels.

The following files are available for the train and test data. Their descriptions are equivalent. 

 * 'train/subject_train.txt': Each row identifies the subject who performed the activity for each window sample. Its range is from 1 to 30. 
 * 'train/Inertial Signals/total_acc_x_train.txt': The acceleration signal from the smartphone accelerometer X axis in standard gravity units 'g'. Every row shows a 128 element vector. The same description applies for the 'total_acc_x_train.txt' and 'total_acc_z_train.txt' files for the Y and Z axis. 
 * 'train/Inertial Signals/body_acc_x_train.txt': The body acceleration signal obtained by subtracting the gravity from the total acceleration. 
 * 'train/Inertial Signals/body_gyro_x_train.txt': The angular velocity vector measured by the gyroscope for each window sample. The units are radians/second. 

*Notes:*

 * *Features are normalized and bounded within [-1,1].*
 * *Each feature vector is a row on the text file.*
 
&nbsp;

## VARIABLES

Here is a description of the data used in this analysis, and that you can find in the dataset.

### ID Fields

 * 1 - activity: chr  
    "laying" "sitting" "standing" "walking" "walkingDownstairs" "walkingUpstairs"
 * 2 - subject: int 
    from 1 to 30

### Extracted Feature Fields

units 'g'

 * 3 - tBodyAccMeanX: num
 * 4 - tBodyAccMeanY: num
 * 5 - tBodyAccMeanZ: num
 * 6 - tBodyAccStdX: num
 * 7 - tBodyAccStdY: num
 * 8 - tBodyAccStdZ: num
 * 9 - tGravityAccMeanX: num
 * 10 - tGravityAccMeanY: num
 * 11 - tGravityAccMeanZ: num
 * 12 - tGravityAccStdX: num
 * 13 - tGravityAccStdY: num
 * 14 - tGravityAccStdZ: num
 * 15 - tBodyAccJerkMeanX: num
 * 16 - tBodyAccJerkMeanY: num
 * 17 - tBodyAccJerkMeanZ: num
 * 18 - tBodyAccJerkStdX: num
 * 19 - tBodyAccJerkStdY: num
 * 20 - tBodyAccJerkStdZ: num
 * 21 - tBodyGyroMeanX: num
 * 22 - tBodyGyroMeanY: num
 * 23 - tBodyGyroMeanZ: num
 * 24 - tBodyGyroStdX: num
 * 25 - tBodyGyroStdY: num
 * 26 - tBodyGyroStdZ: num
 * 27 - tBodyGyroJerkMeanX: num
 * 28 - tBodyGyroJerkMeanY: num
 * 29 - tBodyGyroJerkMeanZ: num
 * 30 - tBodyGyroJerkStdX: num
 * 31 - tBodyGyroJerkStdY: num
 * 32 - tBodyGyroJerkStdZ: num
 * 33 - tBodyAccMagMean: num
 * 34 - tBodyAccMagStd: num
 * 35 - tGravityAccMagMean: num
 * 36 - tGravityAccMagStd: num
 * 37 - tBodyAccJerkMagMean: num
 * 38 - tBodyAccJerkMagStd: num
 * 39 - tBodyGyroMagMean: num
 * 40 - tBodyGyroMagStd: num
 * 41 - tBodyGyroJerkMagMean: num
 * 42 - tBodyGyroJerkMagStd: num
 * 43 - fBodyAccMeanX: num
 * 44 - fBodyAccMeanY: num
 * 45 - fBodyAccMeanZ: num
 * 46 - fBodyAccStdX: num
 * 47 - fBodyAccStdY: num
 * 48 - fBodyAccStdZ: num
 * 49 - fBodyAccJerkMeanX: num
 * 50 - fBodyAccJerkMeanY: num
 * 51 - fBodyAccJerkMeanZ: num
 * 52 - fBodyAccJerkStdX: num
 * 53 - fBodyAccJerkStdY: num
 * 54 - fBodyAccJerkStdZ: num
 * 55 - fBodyGyroMeanX: num
 * 56 - fBodyGyroMeanY: num
 * 57 - fBodyGyroMeanZ: num
 * 58 - fBodyGyroStdX: num
 * 59 - fBodyGyroStdY: num
 * 60 - fBodyGyroStdZ: num
 * 61 - fBodyAccMagMean: num
 * 62 - fBodyAccMagStd: num
 * 63 - fBodyBodyAccJerkMagMean: num
 * 64 - fBodyBodyAccJerkMagStd: num
 * 65 - fBodyBodyGyroMagMean: num
 * 66 - fBodyBodyGyroMagStd: num
 * 67 - fBodyBodyGyroJerkMagMean: num
 * 68 - fBodyBodyGyroJerkMagStd: num

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

## CONTEXT - PROJECT ASSIGNMENT
The purpose of this project is to demonstrate your ability to collect, work with, and clean a data set. The goal is to prepare tidy data that can be used for later analysis. You will be graded by your peers on a series of yes/no questions related to the project. You will be required to submit: 1) a tidy data set as described below, 2) a link to a Github repository with your script for performing the analysis, and 3) a code book that describes the variables, the data, and any transformations or work that you performed to clean up the data called CodeBook.md. You should also include a README.md in the repo with your scripts. This repo explains how all of the scripts work and how they are connected.  

One of the most exciting areas in all of data science right now is wearable computing - see for example this article . Companies like Fitbit, Nike, and Jawbone Up are racing to develop the most advanced algorithms to attract new users. The data linked to from the course website represent data collected from the accelerometers from the Samsung Galaxy S smartphone. A full description is available at the site where the data was obtained: 

<http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones>

&nbsp;

## LICENCE
Use of this dataset in publications must be acknowledged by referencing the following publication [1] 

[1] Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. International Workshop of Ambient Assisted Living (IWAAL 2012). Vitoria-Gasteiz, Spain. Dec 2012

This dataset is distributed AS-IS and no responsibility implied or explicit can be addressed to the authors or their institutions for its use or misuse. Any commercial use is prohibited.

Jorge L. Reyes-Ortiz, Alessandro Ghio, Luca Oneto, Davide Anguita. November 2012.
