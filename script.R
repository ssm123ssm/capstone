# Loading the libraries
if (!require(tidyverse)) install.packages('tidyverse')
if (!require(caret)) install.packages('caret')
if (!require(factoextra)) install.packages('factoextra')
if (!require(broom)) install.packages('broom')
if (!require(readr)) install.packages('readr')
if (!require(pROC)) install.packages('pROC')
if (!require(NeuralNetTools)) install.packages('NeuralNetTools')

library(tidyverse)
library(caret)
library(factoextra)
library(broom)
library(readr)
library(pROC)

#importing the dataset
uri <- 'http://ssm123ssm.github.io/projects/data_arrhythmia.csv'
original <- read_delim(uri,";", escape_double = FALSE, trim_ws = TRUE)


# Filtering the output categories of 1,2,3 and 4
tem <- original %>% filter(diagnosis %in% c('1', '2', '3', '4'))

#Partitioning the dataset to train (70%) and test(30%)
set.seed(2000, sample.kind = 'Rounding')
test_ind <- as.vector(createDataPartition(tem$diagnosis, p = .3, list = F))
train_set <- tem[-test_ind,]
test_set <- tem[test_ind,]

#Upsampling the training set to balance class prevelence
train_set <- upSample(train_set[,-280], as.factor(train_set$diagnosis), list = FALSE)

#Selecting features according to ECG changes during an MI
selected <- c(1,2,15, 161, 162, 167, 171, 172, 177, 181, 182, 187, 191, 192, 197, 201, 202, 207, 211, 212, 217, 221, 222, 227, 231, 232, 237, 241, 242, 247, 251, 252, 257, 261, 262, 267, 271, 272, 277, 16:20, 28:32, 40:44, 52:56, 64:68, 76:80, 88:92, 100:104, 112:116, 124:128, 136:140, 148:152, 280)
selected_names <- c('age', 'sex', 'HR', 'Q_amp_1', 'R_amp_1', 'T_amp_1','Q_amp_2', 'R_amp_2', 'T_amp_2', 'Q_amp_3', 'R_amp_3', 'T_amp_3', 'Q_amp_avR', 'R_amp_avR', 'T_amp_avR', 'Q_amp_avL', 'R_amp_avL', 'T_amp_avL', 'Q_amp_avF', 'R_amp_avF', 'T_amp_avF', 'Q_amp_v1', 'R_amp_v1', 'T_amp_v1', 'Q_amp_v2', 'R_amp_v2', 'T_amp_v2','Q_amp_v3', 'R_amp_v3', 'T_amp_v3', 'Q_amp_v4', 'R_amp_v4', 'T_amp_v4', 'Q_amp_v5', 'R_amp_v5', 'T_amp_v5', 'Q_amp_v6', 'R_amp_v6', 'T_amp_v6', 'Q_wd_1', 'R_wd_1', 'S_wd_1', 'r_wd_1', 's_wd_1', 'Q_wd_2', 'R_wd_2', 'S_wd_2', 'r_wd_2', 's_wd_2', 'Q_wd_3', 'R_wd_3', 'S_wd_3', 'r_wd_3', 's_wd_3','Q_wd_aVR', 'R_wd_aVR', 'S_wd_aVR', 'r_wd_aVR', 's_wd_aVR', 'Q_wd_aVL', 'R_wd_aVL', 'S_wd_aVL', 'r_wd_aVL', 's_wd_aVL','Q_wd_3aVF', 'R_wd_3aVF', 'S_wd_3aVF', 'r_wd_3aVF', 's_wd_3aVF', 'Q_wd_v1', 'R_wd_v1', 'S_wd_v1', 'r_wd_v1', 's_wd_v1', 'Q_wd_v2', 'R_wd_v2', 'S_wd_v2', 'r_wd_v2', 's_wd_v2', 'Q_wd_v3', 'R_wd_v3', 'S_wd_v3', 'r_wd_v3', 's_wd_v3', 'Q_wd_v4', 'R_wd_v4', 'S_wd_v4', 'r_wd_v4', 's_wd_v4', 'Q_wd_v5', 'R_wd_v5', 'S_wd_v5', 'r_wd_v5', 's_wd_v5', 'Q_wd_v6', 'R_wd_v6', 'S_wd_v6', 'r_wd_v6', 's_wd_v6', 'class')
sel_train <-  train_set %>% select(selected)
sel_test <- test_set %>% select(selected)

#Removing Near-Zero variance columns
nz <- nzv(sel_train)
names(sel_train) <- selected_names
names(sel_test) <- selected_names

sel_train <- sel_train[,-nz]
sel_test <- sel_test[,-nz]

#Parsing the predictors to a numeric matrix - training data
tm <- sel_train %>% as.matrix()
tm <- apply(tm, 2, as.numeric)
tm[!is.finite(tm)] = 0
tl <- as.factor(paste0('cl_',tm[,ncol(tm)]))
tm <- tm[,-ncol(tm)]

#Parsing the predictors to a numeric matrix - test data
tmm <- sel_test %>% as.matrix()
tmm <- apply(tmm, 2, as.numeric)
tmm[!is.finite(tmm)] = 0
tll <- as.factor(paste0('cl_',tmm[,ncol(tmm)]))
tmm <- tmm[,-ncol(tmm)]


#using 10 fold cross validation
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,classProbs =  TRUE)

#Training the models, optimized for Kappa after centering and scaling

#treebag
set.seed(2020)
mod_treebag <- train(tm,tl, method = 'treebag' ,metric = "Kappa", preProcess = c("center", "scale"), trControl = fitControl)

#rf
set.seed(2020)
mod_rf <- train(tm,tl, method = 'rf' ,metric = "Kappa", preProcess = c("center", "scale"), trControl = fitControl, tuneGrid = data.frame(mtry = seq(1,10, by = 2)))

#Neural networks
library(NeuralNetTools)
#Multi-Layer Perceptron

set.seed(2020)

set.seed(2020)
mod_mlp <- train(tm,tl, method = 'mlp' ,metric = "Kappa", preProcess = c("center", "scale"), trControl = fitControl, tuneGrid = data.frame(size = 1:10))

#Ploting the parameter tuning
plot(mod_rf, plotType = 'line')
plot(mod_mlp)

#The distributions of the evaluation metrics of the cross validation samples and the scatter plot matrix for Kappa statistic of the three models
resamples <- resamples(list(treebag = mod_treebag, RF = mod_rf, preceptron = mod_mlp))

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)

bwplot(resamples)
dotplot(resamples)

#Confusion matrices of the three models for the predictions for test data
#Random forests
confusionMatrix(tll, predict(mod_rf, tmm))

#Treebag
confusionMatrix(tll, predict(mod_treebag, tmm))

#Perceptron
confusionMatrix(tll, predict(mod_mlp, tmm))



