install.packages("caret", dependencies = c("Depends", "Suggests"))
install.packages("mlbench")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("tree")
install.packages("e1071")

# 1. Prepare Problem
# a) Load libraries
library(mlbench)
library(e1071)
library(lattice)
library(corrplot)
library(caret)
library(klaR)
library(pROC)
library(rpart)
library(rpart.plot)
library(tree)

# b) Load dataset
data<-read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")

# 2. Summarize Data
# a) Descriptive statistics

# peek at data
head(data)

View(data)

# find if any data is unavailable
is.na(data)
anyNA(data)

# see heart-disease.names
names(data) <- c( "age", "sex", "cp", "trestbps", "chol","fbs", "restecg",
                        "thalach","exang", "oldpeak","slope", "ca", "thal", "num")

# data dimension					
dim(data)

str(data)

# get a data summary
summary(data)

sapply(data, class)

# standard deviation
sapply(data[,1:11],sd)

# skewness
skew <- apply(data[,1:11],2,skewness)
print(skew)

# correlations
correlations <- cor(data[,1:11])
print(correlations)

# b) Data visualizations

par("mar")
par(mar=c(0.1,0.1,0.1,0.1))

# histogram 
par(mfrow=c(1,11))
for(i in 1:11) 
    hist(data[,i], main=names(data)[i])

# density plot
for(i in 1:11) 
    plot(density(data[,i], main=names(data)[i]))

# boxplot	
for(i in 1:11) 
    boxplot(data[,i], main=names(data)[i])

# correlation plot
corrplot(correlations, method="circle")

# barplot 
barplot(table(data$num))

plot(data$num)

# 3. Prepare Data
# data cleaning and transforms

# eliminate rows with not found info
which(data$ca %in% c("?"))
data <- data[-c(166, 192, 287,302), ]

# descriptive statistics
# 1 - disease, 0 - no disease
data$num<-ifelse(data$num> 0,"disease","no disease")
# 1 - male, 0 - female
data$sex<-ifelse(data$sex> 0,"male","female")

table(data$sex)

sex_disease<-table(gender=data$sex,disease=data$num)
sex_disease

ggplot(data, aes(x=sex)) + geom_bar(fill="green") + facet_wrap(~num)
ggplot(data, aes(x=num,y=age)) + geom_boxplot()

by(data$age, data$num, summary)

# reset variables
data$num<-ifelse(data$num=="disease",0,1)
data$sex<-ifelse(data$sex=="female",0,1)


# 4. Evaluate Algorithms
# train the data and compare algorithms

# partition into train/test
train_index <- sample(x=1:nrow(data), size=0.8*nrow(data))
train = data[train_index,]
test = data[-train_index,]

# binomial linear model
set.seed(123)
fit<-glm(num~.,data=train, family="binomial")
pred_binomial<-predict(fit,newdata = test,type = "response")
pred_binomial
pred_number_binomial <- ifelse(pred_binomial > 0.5, 1, 0)
predicted_factor_binomial <- factor(pred_number_binomial, levels=c(0, 1))

test$num_factor<-factor(test$num, levels=c(0,1))
conf_matrix_binomial<-confusionMatrix(predicted_factor_binomial,test[,"num_factor"])

conf_matrix_binomial$overall['Accuracy']



# choose model in a stepwise algorithm
stepAIC(fit, direction = "backward")

fit_formula<-glm(formula = num ~ sex + cp + trestbps + restecg + thalach + 
slope +  exang + oldpeak + slope + ca + thal, family = "binomial", 
          data = train)
		  
pred_formula<-predict(fit_formula,newdata = test,type = "response")
pred_formula
pred_number_formula <- ifelse(pred_formula > 0.5, 1, 0)
predicted_factor_formula <- factor(pred_number_formula, levels=c(0, 1))

conf_matrix_formula<-confusionMatrix(predicted_factor_formula,test[,"num_factor"])
conf_matrix_formula

conf_matrix_formula$overall['Accuracy']


# recursive partitioning and classification tree model
fit_rpart <- rpart(formula = num ~ sex + cp + trestbps + restecg + thalach + 
                exang + oldpeak + slope + ca + thal, method = "class", 
              data = train)


# display cross-validation results 
printcp(fit_rpart)
plotcp(fit_rpart) 
summary(fit_rpart) 


# plot tree 

set.seed(123)

fit_rpart <- tree(num ~ sex + cp + trestbps + restecg + thalach + 
               exang + oldpeak + slope + ca + thal, data=train)

fit_rpart

plot(fit_rpart, uniform=TRUE, main="Heart Attack Prediction")
text(fit_rpart, use.n=TRUE, all=TRUE, cex=.8)

# prune the tree 

fit_rpart_pruned<- prune(fit_rpart)

# plot the pruned tree 
plot(fit_rpart_pruned, uniform=TRUE, main="Pruned Classification Tree")

# probability prediction
pred_rpart <- predict(fit_rpart, test)
head(pred_rpart)

# 5. Improve Accuracy
# try to improve test fit:
pruned_tree <- prune.tree(fit_rpart, best=6)
plot(pruned_tree)
text(pruned_tree)
pruned_prediction <- predict(pruned_tree, test, type="vector")
table(pruned_prediction, test$num)

# use cross validation to find the best tree
tree_model <- tree(num ~ ., data=data)
summary(tree_model)
cv_model <- cv.tree(tree_model)
plot(cv_model)
best_size <- cv_model$size[which(cv_model$dev==min(cv_model$dev))]
best_size

cv_model_pruned <- prune.rpart(tree_model,cp=7)
summary(cv_model_pruned)
pruned_prediction <- predict(cv_model_pruned, test, type="vector")
table(pruned_prediction, test$num)
pred_number_pruned <- ifelse(pruned_prediction > 0.5, 1, 0)
predicted_factor_pruned <- factor(pred_number_pruned, levels=c(0, 1))

conf_matrix_pruned<-confusionMatrix(predicted_factor_pruned,test[,"num_factor"])
str(predicted_factor_pruned)

conf_matrix_pruned$overall['Accuracy']


# support vector machine
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3, classProbs = TRUE)
set.seed(123)

svm_model = svm(num ~., data = train, method = "svmLinear",
                  trControl=trctrl,
                  preProcess = c("center", "scale"),
                  tuneLength = 10)

svm_pred = predict(svm_model,test, type="vector")
table(svm_pred==test$num)
pred_number_svm <- ifelse(svm_pred > 0.5, 1, 0)
predicted_factor_svm <- factor(pred_number_svm, levels=c(0, 1))
svm_conf_matrix <- confusionMatrix(predicted_factor_svm,test[,"num_factor"])

svm_conf_matrix$overall['Accuracy']

# 6. Finalize Model
# save models for later use
saveRDS(fit, "./models/binomial.rds")
saveRDS(fit_formula, "./models/formula.rds")
saveRDS(cv_model_pruned, "./models/tree.rds")
saveRDS(svm_model, "./models/svm.rds")

#readRDS("./models/tree.rds")



