#!/usr/bin/env Rscript

## @knitr Practical_Machine_Learning_Project_setup

# set text display width
options(width=80)

# set working directory
setwd('~/Coursera/PracticalMachineLearning/PracticalMachineLearningProject/')

## @knitr Practical_Machine_Learning_Project_section_1

# load data
pml_training_original = read.csv('pml-training.csv')
pml_testing_original = read.csv('pml-testing.csv')

# column names related to belt, arm or dumbbell to be used as predictors
colnames_training = colnames(pml_training_original)
colnames_predictors = colnames_training[ grep(".*belt.*|.*arm.*|.*dumbbell.*",
                                              colnames(pml_training_original)) ]

# eliminate predictor columns with NA values
cols_no_NA = sapply(pml_training_original[,colnames_predictors],
                    function(x) !any(is.na(x)))

# reduce predictors to columns free of NA values
colnames_predictors = colnames_predictors[cols_no_NA]

# eliminate predictor columns with "#DIV/0!"
cols_no_DIV0 = sapply(pml_training_original[,colnames_predictors],
                    function(x) length(grep(".*DIV/0.*",x))==0)
colnames_predictors = colnames_predictors[cols_no_DIV0]

# show number of remaining predictors
length(colnames_predictors)

# dataframes containing only the intended predictors
pml_training = pml_training_original[,colnames_predictors]
pml_training$classe = pml_training_original$classe
pml_testing = pml_testing_original[,c(colnames_predictors)]

## @knitr Practical_Machine_Learning_Project_section_2

# prepare for cross-validation by partitioning provided training data into 
# train and test subsets
require(caret)
trainIndex = createDataPartition(pml_training$classe, p = 0.60, list=FALSE)
pml_training_train = pml_training[ trainIndex,]
pml_training_test  = pml_training[-trainIndex,]

## @knitr Practical_Machine_Learning_Project_section_3

# create a Random Forest model
require(randomForest)
set.seed(20141025)
modelFit = randomForest(classe ~ ., data=pml_training_train)

# display model fit results
modelFit

## @knitr Practical_Machine_Learning_Project_section_4

# Generic plot using model from randomForest
plot(modelFit, log="y",
     main="Estimated Out-of-Bag (OOB) Error and Class Error of Random Forest Model")
legend("top", colnames(modelFit$err.rate), col=1:6, cex=0.8, fill=1:6)

# Dotchart of variable importance as measured by randomForest
varImpPlot(modelFit, main="Variable Importance in the Random Forest Model")

## @knitr Practical_Machine_Learning_Project_section_5

# estimate out of sample error
pred_out_of_sample = predict(modelFit, newdata=subset(pml_training_test, select=-classe))

# out of sample confusion matrix
confusion_matrix = table(pred_out_of_sample, pml_training_test$classe)
confusion_matrix

# estimated out of sample error rate
out_of_sample_error_rate = 1.00 - sum(diag(confusion_matrix)) / sum(confusion_matrix)
out_of_sample_error_rate

## @knitr Practical_Machine_Learning_Project_section_6

# predict using pml-test data
pred = predict(modelFit, newdata=pml_testing)
pred

## @knitr Practical_Machine_Learning_Project_section_7

# write prediction answers to files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_", i, ".txt")
    write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
  }
}
pml_write_files(pred)
