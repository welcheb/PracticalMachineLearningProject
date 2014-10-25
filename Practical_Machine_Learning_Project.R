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

## @knitr Practical_Machine_Learning_Project_section_2

# create a Random Forest model
require(caret)
require(randomForest)
pml_training = pml_training_original[,colnames_predictors]
pml_training$classe = pml_training_original$classe
modelFit = randomForest(classe ~ ., data=pml_training)

## @knitr Practical_Machine_Learning_Project_section_3

# predict using pml test data
pml_testing = pml_testing_original[,c(colnames_predictors)]
pred = predict(modelFit, newdata=pml_testing)
pred

# write answers to files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(pred)
