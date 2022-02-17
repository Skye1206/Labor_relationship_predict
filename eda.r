# path: /home/2021/nyu/fall/xw2113/project/eda.r

## https://www.openml.org/d/246
## https://archive.ics.uci.edu/ml/datasets/Labor+Relations

options (warn = -1)
library(ggplot2)
library(ggthemes)
library(plyr)
library(corrplot)
library(moments)
library(dplyr)
library(randomForest)
library(ROCR)
library(car)
library(caret)

pdf('eda.pdf')

data = data.frame(read.csv('/home/2021/nyu/fall/xw2113/project/data/BNG_labor.csv'))

# if something wrong with the relative path, run the absolute path on IBM Cloud
# data = data.frame(read.csv('/home/2021/nyu/fall/xw2113/project/data/BNG_labor.csv'))
# relative path: data = data.frame(read.csv('data/BNG_labor.csv'))

dim(data)

head(data)

str(data)

##First of all, I would like to see which features contain missing values.

colSums(is.na(data))
NAcol <- which(colSums(is.na(data)) > 0)
print(NAcol)

#checking how many numerical features and how many categorical features.

Charcol <- names(data[,sapply(data, is.character)])
print(Charcol)

Numcol <- names(data[,sapply(data, is.numeric)])
print(Numcol)

#types of cost of living adjustment
table(data$cost.of.living.adjustment)

#Not ordinal, so converting into factors
data$cost.of.living.adjustment <- as.factor(data$cost.of.living.adjustment)
sum(table(data$cost.of.living.adjustment))

#types of pension
table(data$pension)

#Not ordinal, so converting into factors
data$pension <- as.factor(data$pension)
sum(table(data$pension))

#types of education allowance
table(data$education.allowance)

#Ordinal, so label encoding
data$education.allowance <- as.integer(as.character(revalue(data$education.allowance, c('no'=0, 'yes'=1))))
table(data$education.allowance)

sum(table(data$education.allowance))

#types of vacation
table(data$vacation)

#Ordinal, so label encoding
data$vacation <- as.integer(as.character(revalue(data$vacation, c('average'=1, 'below_average'=2, 'generous'=3))))
table(data$vacation)

sum(table(data$vacation))

#types of longterm disability assistance
table(data$longterm.disability.assistance)

#Ordinal, so label encoding
data$longterm.disability.assistance <- as.integer(as.character(revalue(data$longterm.disability.assistance, c('no'=0, 'yes'=1))))
table(data$longterm.disability.assistance)

sum(table(data$longterm.disability.assistance))

#types of contribution to dental plan
table(data$contribution.to.dental.plan)

#Ordinal, so label encoding
data$contribution.to.dental.plan <- as.integer(as.character(revalue(data$contribution.to.dental.plan, c('none'=0, 'half'=1, 'full'=2))))
table(data$contribution.to.dental.plan)

sum(table(data$contribution.to.dental.plan))

#types of bereavement assistance
table(data$bereavement.assistance)

#Ordinal, so label encoding
data$bereavement.assistance <- as.integer(as.character(revalue(data$bereavement.assistance, c('no'=0, 'yes'=1))))
table(data$bereavement.assistance)

sum(table(data$bereavement.assistance))

#types of contribution to health paln
table(data$contribution.to.health.plan)

#Ordinal, so label encoding
data$contribution.to.health.plan <- as.integer(as.character(revalue(data$contribution.to.health.plan, c('none'=0, 'half'=1, 'full'=2))))
table(data$contribution.to.health.plan)

sum(table(data$contribution.to.health.plan))

ggplot(data, aes(x=class, fill = class)) + 
    geom_bar() + 
    theme_wsj() +
    theme(text = element_text(size = 25))

# add a new feature: labor.relation to help eda

data$labor.relation <- ifelse(data$class=='good', "good relation", "bad relation")
data$labor.relation <- as.factor(data$class)

# label encoding the response variable
data$class <- as.integer(as.character(revalue(data$class, c('bad'=0, 'good'=1))))
table(data$class)

str(data)

summary(data)

# The mean wage keeps increase from first year to third year.
# Someone was deducted from their wages in the third year.
# About 70% people do not have cost of living adjustment.
# About 70% employers contribute to their employee's pension.
# The data size is 1000,000, 353000 bad relations and 647000 good relations.


## Question:
#unblanced response variable, under sample?
# Can we add wage.increase.first.year+wage.increase.second.year+wage.increase.third.year to create a new feature?


#split data int train and test

set.seed(43)
index <-  sort(sample(nrow(data), nrow(data)*.8))
train <- data[index,]
test <-  data[-index,]

ggplot(train, aes(x=as.factor(class), fill = as.factor(class))) + 
    geom_bar() + 
    theme_wsj() +
    theme(text = element_text(size = 15))

ggplot(test, aes(x=as.factor(class), fill = as.factor(class))) + 
    geom_bar() + 
    theme_wsj() +
    theme(text = element_text(size = 15))

#downsample
subtrain0 <- train[train$class == 0,] #controls
row.name <- rownames(train[train$class == 1,])

set.seed(43)
resample <- sample(row.name, nrow(subtrain0), replace = T) #resampling
subtrain1 <- train[resample,]  #cases

train <- rbind(subtrain0,subtrain1)

#shuffle
set.seed(42)
rows <- sample(nrow(train))
train <- train[rows, ]

dim(train)

ggplot(train, aes(x=as.factor(class), fill = as.factor(class))) + 
    geom_bar() + 
    theme_wsj() +
    theme(text = element_text(size = 15))

summary(as.factor(train$duration))

ggplot(data=train, aes(x=as.factor(duration))) +
geom_histogram(stat='count', fill = "#60AADD") +
theme(text = element_text(size = 30))

# correlation test
chisq.test(train$labor.relation, as.factor(train$duration),
               correct = F)

## Since we get a p-Value less than the significance level of 0.05, 
## we reject the null hypothesis and conclude that the two variables are in fact dependent.

train %>%
    group_by(labor.relation) %>% 
    summarise(mean = mean(duration), sd = sd(duration),
                    median = median(duration), IQR = IQR(duration))

train %>% 
    ggplot(
        aes(x= as.factor(duration), fill= as.factor(duration))) + 
    geom_bar() +  
    facet_wrap(~ labor.relation) +
    theme(text = element_text(size = 25))

# higher duration is more likely to be a good relation.

summary(train$wage.increase.first.year)

hist(train$wage.increase.first.year, col = "#60AADD")

print(paste("skewness: ", skewness(train$wage.increase.first.year)))
print(paste("kurtosis: ", kurtosis(train$wage.increase.first.year)))

qqnorm(train$wage.increase.first.year)
qqline(train$wage.increase.first.year, col='red')

wilcox.test(train$wage.increase.first.year ~ train$labor.relation, mu=0,
            alternative = "two.sided",
            conf.level= 0.95,
            var.equal = F)

train %>%
    group_by(labor.relation) %>% 
    summarise(mean = mean(wage.increase.first.year), sd = sd(wage.increase.first.year),
                    median = median(wage.increase.first.year), IQR = IQR(wage.increase.first.year))

train %>% 
    ggplot(
        aes(x= wage.increase.first.year, fill= wage.increase.first.year)) + 
    geom_histogram(binwidth = 1, color ="black", fill = "violetred2") + 
    facet_wrap(~ labor.relation)+
    theme(text = element_text(size = 25))

ggplot(train, aes(labor.relation,wage.increase.first.year,fill=labor.relation)) +
geom_boxplot() +
theme(text = element_text(size = 25))

# in the bad relation, most people have a small salary increase in the first year.
# good relation increase salary more than bad relation.

summary(train$wage.increase.second.year)

hist(train$wage.increase.second.year, col = "#60AADD")

print(paste("skewness: ", skewness(train$wage.increase.second.year)))
print(paste("kurtosis: ", kurtosis(train$wage.increase.second.year)))

qqnorm(train$wage.increase.second.year)
qqline(train$wage.increase.second.year, col='red')

wilcox.test(train$wage.increase.second.year ~ train$labor.relation, mu=0,
            alternative = "two.sided",
            conf.level= 0.95,
            var.equal = F)

train %>%
    group_by(labor.relation) %>% 
    summarise(mean = mean(wage.increase.second.year), sd = sd(wage.increase.second.year),
                    median = median(wage.increase.second.year), IQR = IQR(wage.increase.second.year))

# median both are 4

train %>% 
    ggplot(
        aes(x= wage.increase.second.year, fill= wage.increase.second.year)) + 
    geom_histogram(binwidth = 1, color ="black", fill = "violetred2") + 
    facet_wrap(~ labor.relation)+
    theme(text = element_text(size = 25))

ggplot(train, aes(labor.relation, wage.increase.second.year, fill=labor.relation)) +
geom_boxplot() +
theme(text = element_text(size = 25))

summary(train$wage.increase.third.year)

hist(train$wage.increase.third.year, col = "#60AADD")

print(paste("skewness: ", skewness(train$wage.increase.third.year)))
print(paste("kurtosis: ", kurtosis(train$wage.increase.third.year)))

qqnorm(train$wage.increase.third.year)
qqline(train$wage.increase.third.year, col='red')

wilcox.test(train$wage.increase.third.year ~ train$labor.relation, mu=0,
            alternative = "two.sided",
            conf.level= 0.95,
            var.equal = F)

train %>%
    group_by(labor.relation) %>% 
    summarise(mean = mean(wage.increase.third.year), sd = sd(wage.increase.third.year),
                    median = median(wage.increase.third.year), IQR = IQR(wage.increase.third.year))

train %>% 
    ggplot(
        aes(x= wage.increase.third.year, fill= wage.increase.third.year)) + 
    geom_histogram(binwidth = 1, color ="black", fill = "violetred2") + 
    facet_wrap(~ labor.relation)+
    theme(text = element_text(size = 25)) 

ggplot(train, aes(labor.relation, wage.increase.third.year, fill=labor.relation)) +
geom_boxplot() +
theme(text = element_text(size = 25))

summary(train$cost.of.living.adjustment)

# correlation test
chisq.test(train$labor.relation, train$cost.of.living.adjustment,
               correct = F)

ggplot(data=train, aes(cost.of.living.adjustment)) +
geom_histogram(stat='count', fill = "#60AADD") +
theme(text = element_text(size = 25))

train %>% 
    ggplot(
        aes(x= cost.of.living.adjustment, fill= cost.of.living.adjustment)) + 
    geom_bar() +  
    facet_wrap(~ labor.relation)+
    theme(text = element_text(size = 18))

# good relations have more tcf.

summary(train$working.hours)

hist(train$working.hours, col = "#60AADD")

print(paste("skewness: ", skewness(train$working.hours)))
print(paste("kurtosis: ", kurtosis(train$working.hours)))

qqnorm(train$working.hours)
qqline(train$working.hours, col='red')

wilcox.test(train$working.hours ~ train$labor.relation, mu=0,
            alternative = "two.sided",
            conf.level= 0.95,
            var.equal = F)

train %>%
    group_by(labor.relation) %>% 
    summarise(mean = mean(working.hours), sd = sd(working.hours),
                    median = median(working.hours), IQR = IQR(working.hours))

train %>% 
    ggplot(
        aes(x= working.hours, fill= working.hours)) + 
    geom_histogram(binwidth = 3, color ="black", fill = "violetred2") + 
    facet_wrap(~ labor.relation)+
    theme(text = element_text(size = 25))

ggplot(train, aes(labor.relation, working.hours, fill=labor.relation)) + 
geom_boxplot() +
theme(text = element_text(size = 25))

summary(train$pension)

chisq.test(train$pension, train$labor.relation, correct = F)

ggplot(data=train, aes(pension)) +
geom_histogram(stat='count', fill = "#60AADD") +
theme(text = element_text(size = 25)) 

train %>% 
    ggplot(
        aes(x= pension, fill= pension)) + 
    geom_bar() +  
    facet_wrap(~ labor.relation)+
    theme(text = element_text(size = 14))

# good relations have more empl_contr.

summary(train$standby.pay)

hist(train$standby.pay, col = "#60AADD")

print(paste("skewness: ", skewness(train$standby.pay)))
print(paste("kurtosis: ", kurtosis(train$standby.pay)))

wilcox.test(train$standby.pay ~ train$labor.relation, 
                mu=0,
                alternative = "two.sided",
                conf.level= 0.95,
                var.equal = F)

train %>%
    group_by(labor.relation) %>% 
    summarise(mean = mean(standby.pay), sd = sd(standby.pay),
                    median = median(standby.pay), IQR = IQR(standby.pay))

train %>% 
    ggplot(
        aes(x= standby.pay, fill= standby.pay)) + 
    geom_histogram(binwidth = 3, color ="black", fill = "violetred2") + 
    facet_wrap(~ labor.relation)+
    theme(text = element_text(size = 25))

ggplot(train, aes(labor.relation, standby.pay, fill=labor.relation)) + 
geom_boxplot() +
theme(text = element_text(size = 25))

summary(train$shift.differential)

hist(train$shift.differential, col = "#60AADD")

print(paste("skewness: ", skewness(train$shift.differential)))
print(paste("kurtosis: ", kurtosis(train$shift.differential)))

wilcox.test(train$shift.differential ~ train$labor.relation, mu=0,
            alternative = "two.sided",
conf.level= 0.95,
            var.equal = F)

train %>%
    group_by(labor.relation) %>% 
    summarise(mean = mean(shift.differential), sd = sd(shift.differential),
                    median = median(shift.differential), IQR = IQR(shift.differential))

train %>% 
    ggplot(
        aes(x= shift.differential, fill= shift.differential)) + 
    geom_histogram(binwidth = 3, color ="black", fill = "violetred2") + 
    facet_wrap(~ labor.relation)+
    theme(text = element_text(size = 20))

ggplot(train, aes(labor.relation, shift.differential, fill=labor.relation)) + 
geom_boxplot() +
theme(text = element_text(size = 25))

summary(as.factor(train$education.allowance))

chisq.test(train$labor.relation, as.factor(train$education.allowance),
               correct = F)

ggplot(data=train, aes(as.factor(education.allowance))) +
geom_histogram(stat='count', fill = "#60AADD") +
theme(text = element_text(size = 25))

train %>% 
    ggplot(
        aes(x= as.factor(education.allowance), fill= as.factor(education.allowance))) + 
    geom_bar() +
    facet_wrap(~ labor.relation) +
    theme(text = element_text(size = 15))

summary(train$statutory.holidays)

hist(train$statutory.holidays, col = "#60AADD")

print(paste("skewness: ", skewness(train$statutory.holidays)))
print(paste("kurtosis: ", kurtosis(train$statutory.holidays)))

qqnorm(train$statutory.holidays)
qqline(train$statutory.holidays, col='red')

wilcox.test(train$statutory.holidays ~ train$labor.relation, mu=0,
            alternative = "two.sided",
conf.level= 0.95,
            var.equal = F)

train %>%
    group_by(labor.relation) %>% 
    summarise(mean = mean(statutory.holidays), sd = sd(statutory.holidays),
                    median = median(statutory.holidays), IQR = IQR(statutory.holidays))

train %>% 
    ggplot(
        aes(x= statutory.holidays, fill= statutory.holidays)) + 
    geom_histogram(binwidth = 3, color ="black", fill = "violetred2") + 
    facet_wrap(~ labor.relation)+
    theme(text = element_text(size = 25))

ggplot(train, aes(labor.relation, statutory.holidays, fill=labor.relation)) + 
geom_boxplot() +
theme(text = element_text(size = 25))

summary(as.factor(train$vacation))

chisq.test(train$labor.relation, as.factor(train$vacation),
               correct = F)

ggplot(data=train, aes(as.factor(vacation))) +
geom_histogram(stat='count', fill = "#60AADD") +
theme(text = element_text(size = 25))

train %>% 
    ggplot(
        aes(x= as.factor(vacation), fill= as.factor(vacation))) + 
    geom_bar() +  
    facet_wrap(~ labor.relation) +
    theme(text = element_text(size = 25))

summary(as.factor(train$longterm.disability.assistance))

chisq.test(train$labor.relation, as.factor(train$longterm.disability.assistance),
               correct = F)

ggplot(data=train, aes(as.factor(longterm.disability.assistance))) +
geom_histogram(stat='count', fill = "#60AADD") +
theme(text = element_text(size = 20))

train %>% 
    ggplot(
        aes(x= as.factor(longterm.disability.assistance), fill= as.factor(longterm.disability.assistance))) + 
    geom_bar() +  
    facet_wrap(~ labor.relation) +
    theme(text = element_text(size = 15))

summary(as.factor(train$contribution.to.dental.plan))

chisq.test(train$labor.relation, as.factor(train$contribution.to.dental.plan),
               correct = F)

ggplot(data=train, aes(as.factor(contribution.to.dental.plan))) +
geom_histogram(stat='count', fill = "#60AADD") +
theme(text = element_text(size = 20))

train %>% 
    ggplot(
        aes(x= as.factor(contribution.to.dental.plan), fill= as.factor(contribution.to.dental.plan))) + 
    geom_bar() +  
    facet_wrap(~ labor.relation) +
    theme(text = element_text(size = 15))

summary(as.factor(train$bereavement.assistance))

chisq.test(train$labor.relation, as.factor(train$bereavement.assistance),
               correct = F)

ggplot(data=train, aes(as.factor(bereavement.assistance))) +
geom_histogram(stat='count', fill = "#60AADD") +
theme(text = element_text(size = 20))

train %>% 
    ggplot(
        aes(x= as.factor(bereavement.assistance), fill= as.factor(bereavement.assistance))) + 
    geom_bar() +  
    facet_wrap(~ labor.relation) +
    theme(text = element_text(size = 15))

summary(as.factor(train$contribution.to.health.plan))

chisq.test(train$labor.relation, as.factor(train$contribution.to.health.plan),
               correct = F)

ggplot(data=train, aes(as.factor(contribution.to.health.plan))) +
geom_histogram(stat='count', fill = "#60AADD") +
theme(text = element_text(size = 20))

train %>% 
    ggplot(
        aes(x= as.factor(contribution.to.health.plan), fill= as.factor(contribution.to.health.plan))) + 
    geom_bar() +  
    facet_wrap(~ labor.relation) +
    theme(text = element_text(size = 15))

#train$total.wage.increase = train$wage.increase.first.year + train$wage.increase.second.year + train$wage.increase.third.year

str(train)

#remove labor.relation
drop <- c("labor.relation") 
train1 <- train[,!(names(train) %in% drop)]
test1 <- test[,!(names(test) %in% drop)]

set.seed(1206)
quick_RF <- randomForest(x=train1[, 1:16], y=train1$class, ntree=20, importance=TRUE)
imp_RF <- importance(quick_RF)
imp_DF <- data.frame(Variables = row.names(imp_RF), MSE = imp_RF[,1])
imp_DF <- imp_DF[order(imp_DF$MSE, decreasing = TRUE),]

ggplot(imp_DF[1:20,], aes(x=reorder(Variables, MSE), y=MSE, fill=MSE)) + geom_bar(stat = 'identity') + labs(x = 'Variables', y= '% increase MSE if variable is randomly permuted') + coord_flip() + theme(legend.position="none")

# drop wage.increase.third.year: low importance, high corrleation with wage.increase.first.year, 
    #which has higher corrletaion with the response variable and higher importance.
# drop bereavement.assistance, lowest importance, also has high corrleation with longterm.disability.assistance, 
    #which has higher correlation with the response variable and higher importance.

numericVarNames = names(data[,sapply(train, is.numeric)])

train1_numeric <- train1[, names(train1) %in% numericVarNames]
train1_factor <- train1[, !names(train1) %in% numericVarNames]
test1_numeric <- test1[, names(test1) %in% numericVarNames]
test1_factor <- test1[, !names(test1) %in% numericVarNames]

train1_dummy <- as.data.frame(model.matrix(~.-1, train1_factor))
test1_dummy <- as.data.frame(model.matrix(~.-1, test1_factor))

train2 <- cbind(train1_numeric, train1_dummy)
test2 <- cbind(test1_numeric, test1_dummy)

#skewness

# for(i in 1:ncol(train1_numeric)){
#         if (abs(skewness(train1_numeric[,i]))>0.8){
#                 train1_numeric[,i] <- log(train1_numeric[,i] +1)
#         }
# }

# for(i in 1:ncol(test1_numeric)){
#         if (abs(skewness(test1_numeric[,i]))>0.8){
#                 test1_numeric[,i] <- log(test1_numeric[,i] +1)
#         }
# }

cor_numVar <- cor(train2, use="pairwise.complete.obs") #correlations of all numeric variables

cor_sorted <- as.matrix(sort(cor_numVar[,'class'], decreasing = TRUE))
CorHigh <- names(which(apply(cor_sorted, 1, function(x) abs(x)>0)))
cor_numVar <- cor_numVar[CorHigh, CorHigh]

corrplot.mixed(cor_numVar, tl.col="black", tl.pos = "lt")

# remove cost.of.living.adjustmentnone
drop <- c("cost.of.living.adjustmentnone") 
train2 <- train2[,!(names(train2) %in% drop)]
test2 <- test2[,!(names(test2) %in% drop)]

set.seed(43)
model <- lm(class ~., data = train2)
# Make predictions
predictions <- model %>% predict(test2)
# Model performance
data.frame(
  RMSE = RMSE(predictions, test2$class),
  R2 = R2(predictions, test2$class)
)

vif(model)

summary(model)

set.seed(43)
glm_train_2 <- glm(class~.,data=train2,family=binomial())

glm_test <- predict(glm_train_2, test2, type =  "response")

require(ROCR)
glm_auc_1<-prediction(glm_test, test2$class)
glm_prf<-performance(glm_auc_1, measure="tpr", x.measure="fpr")
glm_slot_fp<-slot(glm_auc_1,"fp")
glm_slot_tp<-slot(glm_auc_1,"tp")
glm_fpr3<-unlist(glm_slot_fp)/unlist(slot(glm_auc_1,"n.neg"))
glm_tpr3<-unlist(glm_slot_tp)/unlist(slot(glm_auc_1,"n.pos"))
glm_perf_AUC=performance(glm_auc_1,"auc")
glm_AUC=glm_perf_AUC@y.values[[1]]
print(glm_AUC)


dev.off()
