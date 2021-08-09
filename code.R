library(readr)
library(tidyverse)
library(caret)
library(rattle)
library(corrplot)
library(RColorBrewer)
library(rpart)
library(rpart.plot)


set.seed(1234)

test <- read_csv("pml-testing.csv")
train <- read_csv("pml-training.csv")

str(train)

train1 <- train[colSums(is.na(train))<=0.9*nrow(train)]

train1 <- train1[,8:length(train1)]

testing <- test[colnames(train1)[-53]]

inTrain <- createDataPartition(train1$classe, p=0.8, list = FALSE)

validation <- train1[-inTrain,]
training <- train1[inTrain,]

corMat <- cor(training[,-53])

corrplot(corMat, order = "FPC", method = "color", type = "lower", 
         tl.cex = 0.8, tl.col = rgb(0, 0, 0),mar = c(1, 1, 1, 1), title = "Training Dataset Correlogram")


fit <- trainControl(method='cv', number = 5)

rpart_mod <- train(classe~., data=training, method="rpart", trControl=fit)

fancyRpartPlot(rpart_mod$finalModel)

rpart_prediction <- predict(rpart_mod, newdata = validation) 
rpart_cm <- confusionMatrix(as.factor(validation$classe), rpart_prediction)
rpart_cm

rf_mod <- train(classe~., data=training, method="rf", trControl=fit)
rf_prediction <- predict(rf_mod, newdata = validation)
rf_cm <- confusionMatrix(as.factor(validation$classe), rf_prediction)
rf_cm
plot(rf_mod, main="Accuracy of the model vs predictors")
plot(rf_mod$finalModel, main="Error vs number of trees")


gbm_mod <- train(classe~., data=training, method="gbm", trControl=fit, verbose=F)
gbm_prediction <- predict(gbm_mod, newdata = validation)
gbm_cm <- confusionMatrix(as.factor(validation$classe), gbm_prediction)
gbm_cm
plot(gbm_mod, main="Accuracy of the model vs boosting iterations")


