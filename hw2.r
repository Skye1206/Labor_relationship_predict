if(!require(plyr)) install.packages(c("plyr")) # revalue()
library(plyr)
if(!require(ROCR)) install.packages(c("ROCR")) #ROC
library(ROCR)
if(!require(e1071)) install.packages(c("e1071")) #Naive Bayes & svm
library(e1071)
if(!require(caret)) install.packages(c("caret")) #confusion matrix
library(caret)
if(!require(rpart)) install.packages(c("rpart")) #decision tree
library(rpart)
if(!require(rpart.plot)) install.packages(c("rpart.plot")) #decision tree
library(rpart.plot)
if(!require(class)) install.packages(c("class")) #knn
library(class)
if(!require(parallel)) install.packages(c("parallel")) #parallel
library(parallel)
if(!require(dplyr)) install.packages(c("dplyr")) #move column
library(dplyr)

pdf("hw2.pdf")

data = data.frame(read.csv('/home/2021/nyu/fall/xw2113/project/data/BNG_labor.csv'))

# label encoding from eda.
data$cost.of.living.adjustment <- as.factor(data$cost.of.living.adjustment)
data$pension <- as.factor(data$pension)
data$education.allowance <- as.integer(as.character(revalue(data$education.allowance, c('no'=0, 'yes'=1))))
data$vacation <- as.integer(as.character(revalue(data$vacation, c('average'=1, 'below_average'=2, 'generous'=3))))
data$longterm.disability.assistance <- as.integer(as.character(revalue(data$longterm.disability.assistance, c('no'=0, 'yes'=1))))
data$contribution.to.dental.plan <- as.integer(as.character(revalue(data$contribution.to.dental.plan, c('none'=0, 'half'=1, 'full'=2))))
data$bereavement.assistance <- as.integer(as.character(revalue(data$bereavement.assistance, c('no'=0, 'yes'=1))))
data$contribution.to.health.plan <- as.integer(as.character(revalue(data$contribution.to.health.plan, c('none'=0, 'half'=1, 'full'=2))))
data$class <- as.integer(as.character(revalue(data$class, c('bad'=0, 'good'=1))))

#split data int train and test
set.seed(43)
index <-  sort(sample(nrow(data), nrow(data)*.8))
train <- data[index,]
test <-  data[-index,]
#downsample
subtrain0 <- train[train$class == 0,] #controls
row.name <- rownames(train[train$class == 1,])
set.seed(43)
resample <- sample(row.name, nrow(subtrain0), replace = F)
subtrain1 <- train[resample,]  #cases
train <- rbind(subtrain0,subtrain1)
#shuffle
set.seed(42)
rows <- sample(nrow(train))
train <- train[rows, ]

#feature engineering
numericVarNames = names(data[,sapply(train, is.numeric)])

train_numeric <- train[, names(train) %in% numericVarNames]
train_factor <- train[, !names(train) %in% numericVarNames]
test_numeric <- test[, names(test) %in% numericVarNames]
test_factor <- test[, !names(test) %in% numericVarNames]

train_dummy <- as.data.frame(model.matrix(~.-1, train_factor))
test_dummy <- as.data.frame(model.matrix(~.-1, test_factor))

train <- cbind(train_numeric, train_dummy)
test <- cbind(test_numeric, test_dummy)

# remove cost.of.living.adjustmentnone due to correlation test
drop <- c("cost.of.living.adjustmentnone") 
train <- train[,!(names(train) %in% drop)]
test <- test[,!(names(test) %in% drop)]

train$class <- as.factor(train$class)

#Base Model
start_time <- Sys.time()
log_model <- glm(class~., data=train, family=binomial(link=logit))
end_time <- Sys.time()
log_train_difference <- difftime(end_time, start_time, units='secs')
print(log_train_difference)

summary(log_model)

test.performance <- function(confusion_matrix){
    # performance on test data set.
    print("Performance on test data")
    # accuracy
    accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)

    #precision: p = TP/(TP+FP)
    #recall: r = TP/(TP+FN)
    #true positive rate(same as recall, aka sensitivity): tpr = TP/(TP+FN)
    #false positive rate: fpr = FP/(FP+TN)
    #true negative rate(1-false positive rate, aka specificity): tnr = 1-fpr
    tp = confusion_matrix[2, 2]
    fp = confusion_matrix[1, 2]
    tn = confusion_matrix[1, 1]
    fn = confusion_matrix[2, 1]
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    tnr = 1-fpr

    # f1 score: f1 = 2*p*r/(p+r)
    f1 = 2*p*r/(p+r)

    performance <- matrix(c(accuracy, p, r, fpr, tnr, f1), ncol=1)
    colnames(performance) <- c("Value")
    rownames(performance) <- c("Accuracy","Precision","Recall(sensitivity)", "FPR", "TNR(specificity)", "F1")
    performance <- as.data.frame(performance)
    head(performance, 6)
}

# logistic regression model accuracy on training data.
log_train <- predict(log_model, train[, c(-15)], type =  "response")
log_train_table_mat = table(train$class, log_train > 0.5) 
log_train_accuracy <- sum(diag(log_train_table_mat)) / sum(log_train_table_mat)
print(paste("train accuracy: ", log_train_accuracy))

# logistic regression model accuracy on test data.
start_time <- Sys.time()
log_test <- predict(log_model, test[, c(-15)], type =  "response")
end_time <- Sys.time()
log_pred_difference <- end_time - start_time
print(log_pred_difference)

# confuse matrix
log_test_table_mat = table(test$class, log_test > 0.5) 
print(log_test_table_mat)

log_performance <- test.performance(log_test_table_mat)
print(log_performance)

# ROC plot
log_ROCRpred <- prediction(log_test, test$class)
log_ROCRperf <- performance(log_ROCRpred, 'tpr', 'fpr')
plot(log_ROCRperf, colorize = TRUE, cex.lab=1.5)

# AUC
log_perf_AUC <- performance(log_ROCRpred,"auc")
log_AUC <- log_perf_AUC@y.values[[1]]
print(paste("AUC: ", log_AUC))

# Bootstrapping
set.seed(43)
runModel <- function(df) { glm(class~.,data = df[sample(1:nrow(df),nrow(df),replace=T),], family=binomial(link=logit)) }
lapplyrunmodel <- function(x) runModel(train)
log_models <- lapply(1:10, lapplyrunmodel)

log_preds <- lapply(log_models, FUN = function(M, D=test[,-c(15)]) predict(M, D, type = "response"))
log_preds <- as.data.frame(log_preds)

# preds in all predictions from different models
# nums is the number of models

bias.variance <- function (preds, nums){
    mean_pred <- rowMeans(preds)
    #Bias
    bias = 0
    for (i in 1:length(test$class)){
        bias = bias + (mean_pred[i]-test$class[i])^2
    }
    bias = bias / length(test$class)
    
    #Variance
    var = 0
    for (i in 1:nums){
        for (j in 1:length(test$class)){
            var = var + (preds[,i][j]-mean_pred[i])^2
        }
        var = var / length(test$class)
    }
    var = var / nums
    
    b_v <- matrix(c(bias, var), ncol=1)
    colnames(b_v) <- c("Value")
    rownames(b_v) <- c("Bias","Variance")
    b_v <- as.data.frame(b_v)
    head(b_v)
    
}

log_b_v <- bias.variance(log_preds, 10)
print(log_b_v)

start_time <- Sys.time()
nb_model <- naiveBayes(class~., data=train)
end_time <- Sys.time()
nb_train_difference <- difftime(end_time, start_time, units='secs')
print(nb_train_difference)

summary(nb_model)

# naive bayes model accuracy on training data.
nb_train <- predict(nb_model, train[, c(-15)], type="class")
nb_train_table_mat = table(train$class, nb_train) 
nb_train_accuracy <- sum(diag(nb_train_table_mat)) / sum(nb_train_table_mat)
print(paste("train accuracy: ", nb_train_accuracy))

# performance on test data set.
start_time <- Sys.time()
nb_test_raw <- predict(nb_model, test[, c(-15)], type="raw")
nb_test_class <-unlist(apply(round(nb_test_raw),1,which.max))-1
end_time <- Sys.time()
nb_pred_difference <- end_time - start_time
print(nb_pred_difference)

nb_test_table_mat <- table(test$class, nb_test_class)
print(nb_test_table_mat)

nb_performance <- test.performance(nb_test_table_mat)
print(nb_performance)

# ROC plot
nb_ROCRpred <- prediction(nb_test_raw[, "1"], test$class) #nb_test_raw has two columns: "0": P(x=0), "1": P(x=1)
nb_ROCRperf <- performance(nb_ROCRpred, 'tpr', 'fpr')
plot(nb_ROCRperf, colorize = TRUE, cex.lab=1.5)

# AUC
nb_perf_AUC <- performance(nb_ROCRpred,"auc")
nb_AUC <- nb_perf_AUC@y.values[[1]]
print(paste("AUC: ", nb_AUC))

# Bootstrapping
set.seed(43)
runModel <- function(df) { naiveBayes(class~., data = df[sample(1:nrow(df),nrow(df),replace=T),]) }
lapplyrunmodel <- function(x) runModel(train)
nb_models <- lapply(1:10, lapplyrunmodel)

start_time <- Sys.time()
cl <- makeCluster(3)
clusterExport(cl, c("naiveBayes", "test"), envir=environment())
nb_preds <- parLapply(cl, nb_models, function(M, D=test[,-c(15)]) predict(M, D, type="raw"))
stopCluster(cl)
end_time <- Sys.time()

difference <- difftime(end_time, start_time, units='mins')
print(difference)

nb_preds <- as.data.frame(nb_preds)[,c(2,4,6,8,10,12,14,16,18,20)]
nb_b_v <- bias.variance(nb_preds, 10)
print(nb_b_v)

#split training data set into 11 disjoint sub data sets. Each sub training data sets will have 51316 observations.
sub_trains <- list()
set.seed(43)
train_index <- 1: nrow(train)
each_train <-  round(nrow(train)/11)

for (i in 1:10){
    sub <- sample(train_index, each_train, replace=F)
    sub_trains[[length(sub_trains)+1]] <- sub
    train_index <- setdiff(train_index, sub)
}

#the left observations is the 11 sub training data set.
sub_trains[[length(sub_trains)+1]] <- train_index

start_time <- Sys.time()
cl <- makeCluster(6)
clusterExport(cl, c("svm", "train", "sub_trains"), envir=environment())
svm_models <- parLapply(cl, 1:11,  function(i) svm(class~., data = train[sub_trains[[i]],], probability=TRUE) )
end_time <- Sys.time()
svm_train_difference <- difftime(end_time, start_time, units='hours')
print(svm_train_difference)
#Time difference: 19 mins

# prediction: majority votes
#performance on training data set.

svm_trains <- lapply(1:11,  function(i) predict(svm_models[[i]], train[sub_trains[[i]], c(-15)], probability=TRUE) )
svm_train_probs <- lapply(1:11, function(i) attr(svm_trains[[i]], "probabilities")[,"1"])
svm_train_table_mats <- lapply(1:11, function(i) table(train[sub_trains[[i]],]$class, svm_train_probs[[i]] > 0.5) )
svm_train_accuracies <- lapply(1:11, function(i) sum(diag(svm_train_table_mats[[i]])) / sum(svm_train_table_mats[[i]]) )
svm_print_accuracies <- lapply(1:11, function(i) print(paste("train accuracy for model ", i, ": ", svm_train_accuracies[[i]])) )

# test performance: majority votes.
# use mean probabilities to represent the final 
start_time <- Sys.time()
svm_tests <- lapply(1:11,  function(i) predict(svm_models[[i]], test[, c(-15)], probability=TRUE) )
svm_test_probs <- lapply(1:11, function(i) attr(svm_tests[[i]], "probabilities")[,"1"])
svm_test_classes <- lapply(1:11, function(i) ifelse(svm_test_probs[[i]]>0.5, 1, 0))

getmode <- function(v) {
    uniqv <- unique(v)
    uniqv[which.max(tabulate(match(v, uniqv)))]
}                          
                           
svm_test_class <- vector()
svm_test_prob <- vector()
for (i in 1: length(test$class)){
    temp_class <- vector()
    temp_prob <- vector()
    for (j in 1: 11){
        temp_class <- append(temp_class, svm_test_classes[[j]][[i]])      
        temp_prob <- append(temp_prob, svm_test_probs[[j]][[i]])
    }
    svm_test_class <- append(svm_test_class, getmode(temp_class))
    svm_test_prob <- append(svm_test_prob, mean(temp_prob))
}    

end_time <- Sys.time()
svm_pred_difference <- end_time - start_time
print(svm_pred_difference)

svm_test_table_mat <- table(test$class, svm_test_class)
print(svm_test_table_mat)

svm_performance <- test.performance(svm_test_table_mat)
print(svm_performance)

# ROC plot
svm_ROCRpred <- prediction(svm_test_prob, test$class)
svm_ROCRperf <- performance(svm_ROCRpred, 'tpr', 'fpr')
plot(svm_ROCRperf, colorize = TRUE, cex.lab=1.5)

# AUC
svm_perf_AUC <- performance(svm_ROCRpred,"auc")
svm_AUC <- svm_perf_AUC@y.values[[1]]
print(paste("AUC: ", svm_AUC))
#[1] "AUC:  0.989536392785673"

svm_preds <- as.data.frame(svm_test_probs)
svm_b_v <- bias.variance(svm_preds, 11)
print(svm_b_v)
#[1] "bias:  0.0361489712064151"
#[1] "variance:  0.0307869096625712"

start_time <- Sys.time()
set.seed(43)
dt_model <- rpart(class~., data=train, method="class")
end_time <- Sys.time()
rpart.plot(dt_model, extra = 106, fallen.leaves = T)

dt_train_difference <- difftime(end_time, start_time, units='secs')
print(dt_train_difference)

dt_model

# decision tree model accuracy on training data.
dt_train <- predict(dt_model, train[, c(-15)], type="class")
dt_train_table_mat = table(train$class, dt_train) 
dt_train_accuracy <- sum(diag(dt_train_table_mat)) / sum(dt_train_table_mat)
print(paste("train accuracy: ", dt_train_accuracy))

# performance on test data set.
start_time <- Sys.time()
dt_test_raw <- predict(dt_model, test[, c(-15)], type="prob")
dt_test_class <-unlist(apply(round(dt_test_raw),1,which.max))-1
end_time <- Sys.time()
dt_pred_difference <- difftime(end_time, start_time, units='secs')
print(dt_pred_difference)

dt_test_table_mat <- table(test$class, dt_test_class)
print(dt_test_table_mat)

dt_performance <- test.performance(dt_test_table_mat)
print(dt_performance)

# Since the result from decision tree is not a rank (score), we need to create the score for ROC plot.
# We create the score by using the frequence of positive samples in leaves
# dt_test <- test
# dt_test$score <- NA

# attach(dt_test)
# dt_test$score[pensionnone == 1] = 0.02
# dt_test$score[pensionnone != 1 & longterm.disability.assistance == 0] = 0.07
# dt_test$score[pensionnone != 1 & longterm.disability.assistance != 0 & wage.increase.first.year < 3.2 & education.allowance == 0 & contribution.to.dental.plan < 2] = 0.23
# dt_test$score[pensionnone != 1 & longterm.disability.assistance != 0 & wage.increase.first.year < 3.2 & education.allowance == 0 & contribution.to.dental.plan >= 2] = 0.88
# dt_test$score[pensionnone != 1 & longterm.disability.assistance != 0 & wage.increase.first.year < 3.2 & education.allowance != 0] = 0.80
# dt_test$score[pensionnone != 1 & longterm.disability.assistance != 0 & wage.increase.first.year >= 3.2 & contribution.to.dental.plan < 1 & contribution.to.health.plan >= 2] = 0.21
# dt_test$score[pensionnone != 1 & longterm.disability.assistance != 0 & wage.increase.first.year >= 3.2 & contribution.to.dental.plan < 1 & contribution.to.health.plan < 2] = 0.80
# dt_test$score[pensionnone != 1 & longterm.disability.assistance != 0 & wage.increase.first.year >= 3.2 & contribution.to.dental.plan >= 1] = 0.93
# detach(dt_test)


# ROC plot
dt_ROCRpred <- prediction(dt_test_raw[,"1"], test$class)
dt_ROCRperf <- performance(dt_ROCRpred, 'tpr', 'fpr')
plot(dt_ROCRperf, colorize = TRUE, cex.lab=1.5)

# AUC
dt_perf_AUC <- performance(dt_ROCRpred,"auc")
dt_AUC <- dt_perf_AUC@y.values[[1]]
print(paste("AUC: ", dt_AUC))

# Bootstrapping
start_time <- Sys.time()
set.seed(43)
cl <- makeCluster(3)
runModel <- function(df) { rpart(class~.,data = df[sample(1:nrow(df),nrow(df),replace=T),], method="class") }
lapplyrunmodel <- function(x) runModel(train)

clusterExport(cl, c("runModel", "rpart", "train"), envir=environment())
dt_models <- parLapply(cl, 1:10, lapplyrunmodel)
stopCluster(cl)

end_time <- Sys.time()
difference <- difftime(end_time, start_time, units='mins')
print(difference)

cl <- makeCluster(3)
clusterExport(cl, c("rpart", "test"), envir=environment())
dt_preds <- parLapply(cl, dt_models, function(M, D=test[,-c(15)]) predict(M, D, type="prob"))
stopCluster(cl)

dt_preds <- as.data.frame(dt_preds)[,c(2,4,6,8,10,12,14,16,18,20)]
dt_b_v <- bias.variance(dt_preds, 10)
print(dt_b_v)

start_time <- Sys.time()
set.seed(43)
knn_test <- knn(train = train[, c(-15)], test = test[, c(-15)], cl = train$class, k=10, prob=TRUE)
end_time <- Sys.time()

# pros: KNN does not need to train. cons: Prediction is time consuming.
knn_pred_difference <- difftime(end_time, start_time, units='mins')
print(knn_pred_difference)

knn_test_table_mat = table(test$class, knn_test)
print(knn_test_table_mat)

knn_performance <- test.performance(knn_test_table_mat)
print(knn_performance)

#change knn probability to the probability of sample is class 1
#knn() function in class only returns the highest probability. The original knn probability is the probability of the class knn predict.
#if the predicted class is 0, the probability will be p[x=0], which needs to be changed to p[x=1] = 1-p[x=0].

knn_new_prob <- attr(knn_test,"prob")
for (i in 1:length(knn_new_prob)) {
    if (knn_test[i] == 0) knn_new_prob[i] = 1-knn_new_prob[i]  
}

# ROC plot
knn_ROCRpred <- prediction(knn_new_prob, test$class)
knn_ROCRperf <- performance(knn_ROCRpred, 'tpr', 'fpr')
plot(knn_ROCRperf, colorize = TRUE, cex.lab=1.5)

# AUC
knn_perf_AUC <- performance(knn_ROCRpred,"auc")
knn_AUC <- knn_perf_AUC@y.values[[1]]
print(paste("AUC: ", knn_AUC))

# Bootstrapping
start_time <- Sys.time()
pcl <- makeCluster(10)
set.seed(43)
lapplyrunmodel <- function(x) { 
    sub_train <- train[sample(1:nrow(train),nrow(train),replace=T), ]
    knn(train = sub_train[, c(-15)], test = test[, c(-15)], cl = sub_train$class, k=10, prob=TRUE) 
}
clusterExport(pcl, c("lapplyrunmodel", "knn", "train", "test"), envir=environment())
knn_models <- parLapply(pcl, 1:10, lapplyrunmodel)
stopCluster(pcl)

end_time <- Sys.time()
difference <- difftime(end_time, start_time, units='hours')
print(difference)
# Time difference of 0.8315365 hours

knn_preds <- list()
for (i in 1:10){
    temp_pred <- attr(knn_models[[i]], "prob")
    for (j in 1:length(temp_pred)) {
    if (knn_models[[i]][j] == 0) temp_pred[j] = 1-temp_pred[j]  
    }
    knn_preds[[length(knn_preds)+1]] <- temp_pred
}
knn_preds <- as.data.frame(knn_preds)

knn_b_v <- bias.variance(knn_preds, 10)
print(knn_b_v)

rocs <- plot(log_ROCRperf, cex.lab=1.5, col = "blue")
rocs <- plot(nb_ROCRperf, col ="green", add = TRUE)
rocs <- plot(svm_ROCRperf, col ="black", add = TRUE)
rocs <- plot(dt_ROCRperf, col ="red", add = TRUE)
rocs <- plot(knn_ROCRperf, col ="yellow", add = TRUE)
legend(0.8, 0.3, legend=c("LR", "NB", "SVM", "DT", "KNN"), col=c("blue", "green", "black", "red", "yellow"), lty=1, cex=1.2)

times <- matrix(c(paste(round(log_train_difference[[1]]), units(log_train_difference)), 
                          paste(round(log_pred_difference[[1]]), units(log_pred_difference)),
                          paste(round(nb_train_difference[[1]]), units(nb_train_difference)), 
                          paste(round(nb_pred_difference[[1]]), units(nb_pred_difference)),
                          paste(round(svm_train_difference[[1]]), units(svm_train_difference)),
                          paste(round(svm_pred_difference[[1]]), units(svm_pred_difference)),
                          paste(round(dt_train_difference[[1]]), units(dt_train_difference)),
                          paste(round(dt_pred_difference[[1]]), units(dt_pred_difference)),
                          '/', paste(round(knn_pred_difference[[1]]), units(knn_pred_difference))), nrow=2)
colnames(times) <- c("LR", "NB", "SVM", "DT", "KNN")
rownames(times) <- c("training time", "testing time")
print(times)

performances <- cbind(log_performance, nb_performance, svm_performance, dt_performance, knn_performance)
performances <- rbind(performances, c(log_AUC, nb_AUC, svm_AUC, dt_AUC, knn_AUC))
colnames(performances) <- c("LR", "NB", "SVM", "DT", "KNN")
rownames(performances)[7] <- c("AUC")
print(performances)

bvs <- cbind(log_b_v, nb_b_v, svm_b_v, dt_b_v, knn_b_v)
colnames(bvs) <- c("LR", "NB", "SVM", "DT", "KNN")
total_error <- bvs["Bias", ] + bvs["Variance", ]
bvs <- rbind(bvs, total_error)
rownames(bvs)[3] <- c("Total Error")
print(bvs)

log_class <- ifelse(log_test > 0.5,1,0) 
log_class <- as.data.frame(log_class)
nb_class <- as.data.frame(nb_test_class)
colnames(nb_class) <- c("nb_class")
svm_class <- as.data.frame(svm_test_class)
colnames(svm_class) <- c("svm_class")
dt_class <- as.data.frame(dt_test_class)
colnames(dt_class) <- c("dt_class")
knn_class <- as.data.frame(knn_test)
colnames(knn_class) <- c("knn_class")
result <- cbind(test, log_class, nb_class, svm_class, dt_class, knn_class)
result <- result %>% relocate(class, .after = pensionret_allw)
head(result)

dev.off()
