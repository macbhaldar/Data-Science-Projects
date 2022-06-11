library(tidyverse)

data <- read.csv("heart.csv")
head(data)

glimpse(data)

ncol(data)
nrow(data)

colnames(data)

summary(data)

# Data Transformation
data2 <- data %>%
  mutate(sex = if_else(sex == 1, "MALE", "FEMALE"),
         fbs = if_else(fbs == 1, ">120", "<=120"),
         exang = if_else(exang == 1, "YES" ,"NO"),
         cp = if_else(cp == 1, "ATYPICAL ANGINA",
                      if_else(cp == 2, "NON-ANGINAL PAIN", "ASYMPTOMATIC")),
         restecg = if_else(restecg == 0, "NORMAL",
                           if_else(restecg == 1, "ABNORMALITY", "PROBABLE OR DEFINITE")),
         slope = as.factor(slope),
         ca = as.factor(ca),
         thal = as.factor(thal),
         target = if_else(target == 1, "YES", "NO")
  ) %>% 
  mutate_if(is.character, as.factor) %>% 
  dplyr::select(target, sex, fbs, exang, cp, restecg, slope, ca, thal, everything())

# Data Visualization
library(repr)
options(repr.plot.width =6, repr.plot.height =3)

# Bar Plot for target (heart disease)
ggplot(data2, aes(x=data2$target, fill=data2$target))+
  geom_bar()+
  xlab("Heart Disease")+
  ylab("count")+
  ggtitle("Presence & Absence of Heart Disease")+
  scale_fill_discrete(name= 'Heart Disease', labels =c("Absence", "Presence"))

prop.table(table(data2$target))

# count the frequency of the values of age
data2 %>%
  group_by(age) %>%
  count() %>%
  filter(n>10) %>%
  ggplot()+
  geom_col(aes(age, n), fill = 'green')+
  ggtitle("Age Analysis")+
  xlab("Age")+
  ylab("Agecount")

# comapre blood pressure across the chest pain
data2 %>%
  ggplot(aes(x=sex, y=trestbps))+
  geom_boxplot(fill ='purple')+
  xlab('sex')+
  ylab('BP')+
  facet_grid(~cp)

data %>%
  ggplot(aes(x=sex, y=trestbps))+
  geom_boxplot(fill ='purple')+
  xlab('sex')+
  ylab('BP')+
  facet_grid(~cp)

data2 %>%
  ggplot(aes(x=sex, y=chol))+
  geom_boxplot(fill ='orange')+
  xlab('sex')+
  ylab('Chol')+
  facet_grid(~cp)

# Correlation

library(corrplot)
library(ggplot2)

cor_heart <- cor(data2[, 10:14])
cor_heart
corrplot(cor_heart, method ='square', type='upper')

??method

s = sum(is.na(data2))
s

is.null(data2)

# Training and Testing the data
library(caret)
set.seed(10)

colnames(data2)

inTrainRows <- createDataPartition(data2$target,p=0.7,list=FALSE)
trainData <- data2[inTrainRows,]
testData <-  data2[-inTrainRows,]
nrow(trainData)/(nrow(testData)+nrow(trainData)) #check the percentage

AUC = list()
Accuracy = list()

set.seed(10)
logRegModel <- train(target ~ ., data=trainData, method = 'glm', family = 'binomial')
logRegPrediction <- predict(logRegModel, testData)
logRegPredictionprob <- predict(logRegModel, testData, type='prob')[2]
logRegConfMat <- confusionMatrix(logRegPrediction, testData[,"target"])
#ROC Curve
library(pROC)
AUC$logReg <- roc(as.numeric(testData$target),as.numeric(as.matrix((logRegPredictionprob))))$auc
Accuracy$logReg <- logRegConfMat$overall['Accuracy']  #found names with str(logRegConfMat)  

# Support Vector Machine
set.seed(10)
svmModel <- train(target ~ ., data = trainData,
                  method = "svmRadial",
                  trControl = fitControl,
                  preProcess = c("center", "scale"),
                  tuneLength = 8,
                  metric = "ROC")
svmPrediction <- predict(svmModel, testData2)
svmPredictionprob <- predict(svmModel, testData2, type='prob')[2]
svmConfMat <- confusionMatrix(svmPrediction, testData2[,"target"])
#ROC Curve
AUC$svm <- roc(as.numeric(testData2$target),as.numeric(as.matrix((svmPredictionprob))))$auc
Accuracy$svm <- svmConfMat$overall['Accuracy']

install.packages('kernlab')

fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

set.seed(10)
svmModel <- train(target ~ ., data = trainData,
                  method = "svmRadial",
                  trControl = fitControl,
                  preProcess = c("center", "scale"),
                  tuneLength = 8,
                  metric = "ROC")
svmPrediction <- predict(svmModel, testData)
svmPredictionprob <- predict(svmModel, testData, type='prob')[2]
svmConfMat <- confusionMatrix(svmPrediction, testData[,"target"])
#ROC Curve
AUC$svm <- roc(as.numeric(testData$target),as.numeric(as.matrix((svmPredictionprob))))$auc
Accuracy$svm <- svmConfMat$overall['Accuracy']

# Random Forest
library(randomForest)
set.seed(10)
RFModel <- randomForest(target ~ .,
                        data=trainData, 
                        importance=TRUE, 
                        ntree=200)
#varImpPlot(RFModel)
RFPrediction <- predict(RFModel, testData)
RFPredictionprob = predict(RFModel,testData,type="prob")[, 2]

RFConfMat <- confusionMatrix(RFPrediction, testData[,"target"])

AUC$RF <- roc(as.numeric(testData$target),as.numeric(as.matrix((RFPredictionprob))))$auc
Accuracy$RF <- RFConfMat$overall['Accuracy']

# Comparison of AUC and Accuracy
row.names <- names(Accuracy)
col.names <- c("AUC", "Accuracy")
cbind(as.data.frame(matrix(c(AUC,Accuracy),nrow = 3, ncol = 2,
                           dimnames = list(row.names, col.names))))

summary(logRegModel)$coeff

'''
How do you find z score in R?
Z= (value – mean)/ (Standard Deviation)

In general, the z score tells you how far a value is from the average of the data in terms of standard deviations.
'''

'''
The standard score (more commonly referred to as a z-score) is a very useful statistic because it
(a) allows us to calculate the probability of a score occurring within our normal distribution and
(b) enables us to compare two scores that are from different normal distributions.
'''

# Confusion Matrix
logRegConfMat

RFConfMat

svmConfMat

'''
p value to test hypothesis: smaller the p value, reject the proposed hypothesis and accepty the alternative one.
'''

'''
A comparison of the area under the ROC and the accuracy of the model predictions shows that logistic regression performs best 
(accuracy of 0.87). Tree-based methods shows low accuracy.
'''
