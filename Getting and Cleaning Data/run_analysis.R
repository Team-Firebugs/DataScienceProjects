# run_analysis


# Step1 - Merges the training and the test sets to create one data set.

# First we read the different txt files and we get sure that the dimensions
# corresponds to what we expect.
X_train <- read.table("./UCI HAR Dataset/train/X_train.txt")
dim(X_train) # 7352 * 561
subject_train <- read.table("./UCI HAR Dataset/train/subject_train.txt")
dim(subject_train) # 7352 * 1
y_train <- read.table("./UCI HAR Dataset/train/y_train.txt")
dim(y_train) # 7352 * 1

X_test <- read.table("./UCI HAR Dataset/test/X_test.txt")
dim(X_test) # 2947 * 561
subject_test <- read.table("./UCI HAR Dataset/test/subject_test.txt")
dim(subject_test) # 2947 * 1
y_test <- read.table("./UCI HAR Dataset/test/y_test.txt")
dim(y_test) # 2947 * 1

# Then We merge the training and the test datasets
mergeData <- rbind(X_train, X_test)
dim(mergeData) # 10299 * 561
mergeLabel <- rbind(y_train, y_test)
dim(mergeLabel) # 10299 * 1
names(mergeLabel) <- "activity_label" # naming the title of the column
mergeSubject <- rbind(subject_train, subject_test)
dim(mergeSubject) # 10299 * 1
names(mergeSubject) <- "Subject" # naming the title of the column

# We have now merged the training and the test. We still have to complete 
# the dataset by adding the Subject column and the Label column. We'll do this 
# after the step 2 so that it's easier to remove all the non-mean and 
# non-standard measuring columns.


# Step2 - Extracts only the measurements on the mean and standard deviation for
# each measurement.

# We first read the file containing the features names
features_name <- read.table("./UCI HAR Dataset/features.txt", sep = " ", col.names = c("ID_features", "Desc_feature"))
dim(features_name) # 561 * 2

# Here we get the indices for all the features with mean and std measurements
indices <- grep("[Mm]ean\\(.*\\)|[Ss]td\\(.*\\)", features_name$Desc_feature)
length(indices) # 66
# [Mm]ean\\(.*\\) means we are extracting the word "mean" or "Mean" with
# parentheses after. That means that we don't want features where the word
# "mean" is inside the parentheses.
# [Ss]td\\(.*\\) the same as above but whith the word "std" or "Std" this time.
# | means we are asking for either the first condition OR the other one.

# Then we make the new dataset by selecting only the features we're interested in
# and that we have selected just before
mergeData_new <- mergeData[, indices]
dim(mergeData_new) # 10299 * 66
features_name_new <- features_name[indices,]
dim(features_name_new) # 66 * 2

# now, let's add the Label and the Subject columns to the dataset. But before, 
# we are going to put features names on top of the columns of the set.

names(mergeData_new) <- features_name_new[,2] # Changing the names

mergeData_new$subject <- mergeSubject[,1] # adding the subject column by merging


# Step 3 - Uses descriptive activity names to name the activities in the data set

# Here, we first read the activity labels dataset
# Then we substitute the "-" by "" in the labels
# Finally, for the 2 labels which are composed of 2 words, we change the first
# letter of the second word by an uppercase
activity_label <- read.table("./UCI HAR Dataset/activity_labels.txt", sep = " ", col.names = c("ID_activity", "activity"))
activity_label[,2] <- tolower(gsub("_","",activity_label[,2]))
substr(activity_label$activity[2], 8, 8) <- toupper(substr(activity_label$activity[2], 8, 8))
substr(activity_label$activity[3], 8, 8) <- toupper(substr(activity_label$activity[3], 8, 8))

mergeData_new$activity_label <- mergeLabel[,1] # adding the activity column by merging
# We then add the activity column to the dataset by merging
mergeDataFinal <- merge(mergeData_new,activity_label, by.x = "activity_label", by.y = "ID_activity", all = TRUE)

# deleting the first column of the dataset which contains the number associated with the activity, which we don't need because we've add the activity name.
mergeDataFinal <- mergeDataFinal[,2:ncol(mergeDataFinal)]


# Step 4 - Appropriately labels the data set with descriptive variable names.

names(mergeDataFinal) <- gsub("\\(\\)","",names(mergeDataFinal)) # delete parentheses
names(mergeDataFinal) <- gsub("mean","Mean",names(mergeDataFinal)) # replacing mean by Mean
names(mergeDataFinal) <- gsub("std","Std",names(mergeDataFinal)) # replacing std by Std
names(mergeDataFinal) <- gsub("-","",names(mergeDataFinal)) # deleting -

mergeDataFinal <- mergeDataFinal[,c(67,68,1:66)] # reordering of the data columns

# Step 5 - From the data set in step 4, creates a second, independent tidy data 
# set with the average of each variable for each activity and each subject.

# Here, we create a matrix to present the data.
# As we compute the average for each activity and each subject, the matrix must
# have 180 rows = 6 * 30 (6 activities and 30 subjects).
# The number of columns is equal to the number of features.
mat <- aggregate(.~ activity + subject, data = mergeDataFinal, mean)
mat <- mat[order(mat$subject,mat$activity),]

write.table(mat, file = "tidyDataSet.txt", row.name = FALSE)
