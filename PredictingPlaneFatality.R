#Name: Sam Tucker (created and ran code for Predictive Modeling)
#Name: Nathan Schleisman (created and ran code for Lasso and Ridge Regression)
#Name: Saul Varshavsky (created and ran code for Descriptive Modeling)
#Name: Hunter Hildebrand (checked and evaluated code and comments and created
#data visualizations)

rm(list = ls()) #to clear workspace

library(ggplot2) #for professional exploratory graphics
library(pROC) #for plotting ROC curves
library(RColorBrewer) #color palettes
library(randomForest) #to fit random forest
library(dplyr) #for piping
library(glmnet) #for lasso regression

### DATA PREPARATION FOR RANDOM FOREST (Sam) -----

#read in the csv data file portraying various metrics when analyzing previous
#plane accidents (Source: https://www.kaggle.com/code/abilashcheruvathur/airplane-accident-severity-analysis-prediction/data)
aa <- read.csv("Datasets/test (1).csv", stringsAsFactors = TRUE)

#We want to check for missing data
unique(is.na(aa))
#Since all the columns returned FALSE, we know our data is clean and there
#are no missing values.

#Let's look at all the columns of the dataset to verify whether the
#intended dataset (a.k.a csv file) was read in correctly
str(aa)
#Based on the names of the columns, this verifies that the dataset was
#read in correctly

#Since the columns Accident_ID, X, and Accident_Type_Code hold no value to us,
#we will drop these columns from the dataset
aa2 <- subset(aa, select = -c(Accident_ID, X, Accident_Type_Code))

#Let's verify whether the above columns were indeed dropped
str(aa2)
#Since we don't see the names of the columns we dropped anymore, this is 
#verification that those columns were indeed dropped successfully

#Let's look at the above line of code again to see if all the variables have the
#appropriate data types associated with them
str(aa2)
#Based on the definitions of each of the variables in our project proposal,
#the data types are appropriate for all of them. Since all of the
#explanatory variables contain numbers as their values, it would make sense
#for each of those columns to be numeric or integers if there are no decimals.
#Additionally, since our predictor variable Severity contains several categories,
#it would make sense for that variable to hold factor data types (a.k.a 
#stringsAsFactors was specified as true when reading in the dataset)

#With regards to the predictor variable Severity, our goal for this
#project is to make predictions as to whether an accident is fatal or not, where
#1 represents fatality and 0 represents otherwise. Therefore, it's best
#to make the variable Severity binary. 

#Before we can make the variable binary, let's determine all the categories
#present in that variable
unique(aa2$Severity)
#Based on the results from running the above line of code, Severity has the
#following categories:
#Highly_Fatal_And_Damaging
#Significant_Damage_And_Fatalities
#Significant_Damage_And_Serious_Injuries
#Minor_Damage_And_Injuries

#Hence, the categories Highly_Fatal_And_Damaging and
#Significant_Damage_And_Fatalities are the categories that should be converted to
#1, since they include the words "Fatal" and "Fatalities". On the other hand,
#the other categories should be converted to 0, since they don't include any
#kind of word that has to do with fatality or death.

#Now we are ready to convert Severity into a binary variable using the
#information described above
for(i in 1:nrow(aa2)) {
  if(aa2$Severity[i] == "Highly_Fatal_And_Damaging") {
    aa2$Severity_Binary[i] = "Yes"
  }
  else if(aa2$Severity[i] == "Significant_Damage_And_Fatalities") {
    aa2$Severity_Binary[i] = "Yes"
  }
  else {
    aa2$Severity_Binary[i] = "No"
  }
}
#The above for loop goes through every single row in the dataset and checks to
#see the category/value for the Severity variable corresponding to
#that row. If the Severity variable corresponding to that row contains the
#words "Fatal" or "Fatalities", then the value of 1 will be assigned to a new
#variable called Severity_Binary, which will be our new predictor variable.
#If the Severity variable corresponding to a given row in the dataset doesn't
#have values containing the words "Fatal" or "Fatalities", then Severity_Binary
#will be assigned a value of 0 for that given row.
#Note: Even though Severity_Binary is a binary variable, it is nevertheless
#actually 
#assigned the values of either "Yes" or "No", because character/factor 
#variables are
#needed when making a random forest model.

#Since we specified stringsAsFactors as true for the dataset, 
#let's convert the values of Severity_Binary from character to factor
aa2$Severity_Binary <- as.factor(aa2$Severity_Binary)

#Let's verify whether Severity_Binary is indeed a binary factor column 
#of values containing either "Yes" or "No"
unique(aa2$Severity_Binary)
#Based on the above line of code, Severity_Binary indeed can take on only two
#values, either "Yes" or "No"

str(aa2$Severity_Binary)
#Based on the above line of code, Severity_Binary is indeed a factor variable.
#Therefore, this fully verifies that Severity_Binary is the way it is intended
#to be.

#Since the Severity_Binary column was created successfully as intended, we
#can now drop the column Severity since it won't be our predictor variable
#anymore.
aa3 <- subset(aa2, select = -c(Severity))
str(aa3)
#We have verified that the column Severity was indeed dropped

### FITTING A BASELINE RANDOM FOREST (Sam) ----

#Before constructing any random forest, it's first important to set the seed
#and split the data into training and testing data

#set the seed
RNGkind(sample.kind = "default")
set.seed(2291352)

#split data into training and testing data
#Note: 80% of training data and 20% of testing data
train.idx <- sample(x = 1:nrow(aa3), size = .8*nrow(aa3))
train.df <- aa3[train.idx,]
test.df <- aa3[-train.idx,]

#Now we are ready to fit a baseline random forest
#But first, let's check to see how many explanatory variables we have
str(aa3)
#We have 9 explanatory variable available. Therefore, we will sample sqrt(9)
#x's for each tree in the baseline forest
myforest <- randomForest(Severity_Binary ~ . ,
                         data = train.df,
                         ntree = 1000,
                         mtry = 3, #sqrt(9) as number of x's to sample for each tree
                         importance = TRUE)

#Let's look at the confusion matrix of the forest
myforest
#Using the information displayed, let's determine the overall accuracy of the
#forest: (true positive + true negative)/(all values in the confusion matrix)
(842 + 1004)/(842 + 1004 + 44 + 110)
#Based on the calculation above, we yielded an accuracy of 0.923.
#In other words, the baseline forest is 92.3% accurate.

#Now, let's determine the OOB error by doing 1 minus the accuracy computed
#above
1 - ((842 + 1004)/(842 + 1004 + 44 + 110))
#We yielded an OOB error of 0.077, which in percentage form can be expressed
#as 7.7%

### TUNING THE BASELINE RANDOM FOREST (Sam) -----

#Although the OOB error is not very high for the random forest, it would still
#be best practice to tune it to see if the OOB error can become even smaller

#A key parameter than we can tune is m, the number of explanatory variables
#available to be sampled for each tree

#In order to tune m, we must consider all of our explanatory variables, which
#in this case is 9
mtry <- c(1:9)

#Now, the process of tuning m begins:

#1. making empty data frame for m and OOB error rate
keeps <- data.frame(m = rep(NA, length(mtry)),
                    OOB_error_rate = rep(NA, length(mtry)))

#2. fit forests with different values for mtry to tune mtry.
for (idx in 1:length(mtry)){
  
  print(paste0("Fitting m = ", mtry[idx]))
  
  tempforest <- randomForest(Severity_Binary ~ . ,
                             data = train.df,
                             ntree = 1000,
                             mtry = mtry[idx]) #mtry is varying with idx
  
  
  #record B and the corresponding OOB error
  keeps[idx, "m"] <- mtry[idx]
  keeps[idx, "OOB_error_rate"] <-mean(predict(tempforest) != train.df$Severity_Binary)
  
}

#3. checking if keeps filled correctly
keeps
#Yes, keeps is indeed filled completely with values indicating 
#OOB error rates

#4. plot m vs OOB error note (check to see if there are errors earlier in this
#code as this graph looks odd)
ggplot(data = keeps) +
  geom_line(aes(x = m, y = OOB_error_rate)) +
  scale_x_continuous(breaks = c(1:9)) +
  labs(x = "m (mtry): # of x variables sampled",
       y = "OOB Error Rate")
#Interestingly enough, when m = 9 (which is the
#greatest m can possibly be), our OOB error is the smallest.

### FITTING A FINAL TUNED RANDOM FOREST (Sam) ----

#Using the information just obtained above from tuning our baseline
#random forest, let's fit our final tuned random forest
finalForest <- randomForest(Severity_Binary ~ .,
                            data = train.df,
                            ntree = 1000,
                            mtry = 9, #based on tuning above
                            importance = TRUE)
#As shown in the above code, mtry is changed to 9 since an mtry (or m) 
#of 9 yields
#the smallest OOB error

#Let's look at the finalForest and compute it's new accuracy and OOB error
finalForest

#accuracy
(1021+909)/(1021+909+27+43)
#0.965 (our final forest is about 97% accurate)
#This is definitely an improvement. As mentioned previously, the accuracy
#of our baseline forest was 92.3%; 97% is definitely greater than 92.3%

#OOB error
1-((1021+909)/(1021+909+27+43))
#0.035 (our final forest has a 3.5% OOB error)
#Therefore, it also makes sense that our OOB error for the final forest
#is smaller than our OOB error for our baseline forest
#(3.5% OOB error compared to the previous 7.7% OOB error)

### PREDICTIVE MODELING RESULTS (Sam) -----

#Let's plot an ROC Curve. However, before that is done, let's create a new
#variable called pi_hat that stores the probabilities of all the positive
#events (which in this case are the events of 1 -- where a fatality
#occurred)
pi_hat <- predict(finalForest, test.df, type = 'prob')[,"Yes"]

#Plot ROC curve
rocCurve <- roc(response = test.df$Severity_Binary,
                predictor = pi_hat,
                levels = c("No","Yes"))
plot(rocCurve, print.thres = TRUE, print.auc = TRUE)
#Based on the results above, if we set out threshold at 0.547 (pi_star), 
#we are guaranteed
#a specificity of 0.975 and sensitivity of 0.977. In other words,
#when a plane crash actually has fatalities, our final random forest model
#correctly predicts that there are indeed fatalities 97.7% (or about 98%) of
#the time. Additionally, When a plane crash doesn't actually have any
#fatalities, our final random forest model correctly predicts no
#fatalities 97.5% (about also 98%) of the time.
#AUC (a.k.a Area under the graph's curve) is 0.996

#Finally, it would be helpful for us to know which variables are the most
#important for our final random forest model.
varImpPlot(finalForest, type = 1)
#Based on the results from the graph, the variables Safety_Score, 
#Days_Since_Inspection, and
#Control_Metric
#are the three most important for this model, because removing them
#would result in a major decrease in the accuracy of the model. A 
#decreased accuracy would hereby result in a greater increase
#in the OOB error for the model (and we want to minimize the OOB error
#as much as possible).

### DATA PREPARATION FOR A GLM MODEL (Saul) ------

#As described above, the predictor variable Severity_Binary is binary, as
#there will only be two values assigned: 1 if a fatality occurs and 0
#otherwise.

#Therefore, let's fit a Bernoulli GLM as our descriptive model.

#Before we fit a Bernoulli GLM, let's convert the Severity_Binary value
#of "Yes" to 1 and the value of "No" to 0. Unlike a random forest model,
#the predictor variable must hold numeric values in order for the Bernoulli
#GLM to work.
aa3$Severity_Binary_2 <- ifelse(aa3$Severity_Binary == "Yes", 1, 0)
#The above line of code created another column called Severity_Binary_2 that
#uses 1 to represent "Yes" from Severity_Binary and 0 to represent "No" from
#Severity_Binary. It's important to note that the variable Severity_Binary_2
#is specifically designed to be used for the GLM model whereas the
#Severity_Binary variable was specifically designed to be used for the
#Random Forest model from above.

#Let's verify that the variable Severity_Binary_2 was actually created and
#only holds the values of 1 or 0
str(aa3)
#There is indeed a newly created variable now called Severity_Binary_2 that is
#indeed numeric as intended
unique(aa3$Severity_Binary_2)
#The variable Severity_Binary_2 indeed can only take on the value of 1 or 0

#Now, we are ready to fit a baseline Bernoulli GLM where Severity_Binary_2 is
#our predictor variable
#Note: The link function of our Bernoulli GLM will be a logit link, since it
#is the best link function when it comes to making meaningful and
#understandable interpretations pertaining to our model.

#Our baseline model will only start with one variable for now. Then, as we
#start to add more variables to the model and the AIC decreases, that is a sign
#that the more variables that began to be added need to be kept in the model.
#As soon as adding more variables to the model results in a greater AIC, then
#that is a sign that no more variables need to be added to the model.
#Note: The variables first included in the model will be those determined to be 
#the
#most important from the Variable Importance Plot

#First, let's start out with only the most important variable from above, which
#is Safety_Score since removing it would result in the greatest decrease in
#mean accuracy (a.k.a greatest OOB error)

m1 <- glm(Severity_Binary_2 ~ Safety_Score, data = aa3, 
          family = binomial(link = "logit"))
AIC(m1)
#The AIC of m1 is 3341.747. Let's add the next most
#important variable and see if the AIC begins
#to decrease.

#In this context, the next most important variable is Days_Since_Inspection for
#similar reasoning as above for the Safety_Score variable.
m2 <- glm(Severity_Binary_2 ~ Safety_Score + Days_Since_Inspection, data = aa3, 
          family = binomial(link = "logit"))
AIC(m2)
#The AIC of m2 is 3203.88 which is less than the AIC of m1 at
#3341.747. Therefore, let's keep both of these variables in the model.

#Now let's fit the next most important variable in the model, which is
#Control_Metric
m3 <- glm(Severity_Binary_2 ~ Safety_Score + Days_Since_Inspection + 
            Control_Metric, data = aa3, 
          family = binomial(link = "logit"))
AIC(m3)
#The AIC of m3 is 3190.479 which is less than the AIC of both m1 and m2.
#This is definitely a sign that all three of those variables should be kept.

#Finally, let's add the variable Adverse_Weather_Metric and see how that affects
#the model's AIC.
m4 <- glm(Severity_Binary_2 ~ Safety_Score + Days_Since_Inspection + 
            Control_Metric + Adverse_Weather_Metric, data = aa3, 
          family = binomial(link = "logit"))
AIC(m4)
#The AIC of m4 now turns out to be slightly greater (3190.63) than the
#AIC of m3 (3190.479). Therefore, let's stick with m3 since the AIC increased
#when fitting m4.

### CHECKING FOR COMPLETE SEPARATION (Saul) -----

#However, we can't yet reasonably conclude that m3 is our final Bernoulli
#GLM model, because we need to check if there is evidence of complete
#separation.

#Since all of our explanatory variables have numeric/integer data types, let's
#first analyze the scatter plots between the explanatory variables and
#the predictor variable to see if there are any indications of complete
#separation.

#scatter plot of Safety_Score vs. Severity_Binary_2
ggplot(data = aa3, aes(x = Safety_Score, y = Severity_Binary_2)) + 
  geom_point(alpha = 0.4, size = 1, position = position_jitter(width = 10, height = 0.1)) + 
  labs(x = "The Safety Score Indicating How Safe The Plane Was Deemed To Be", 
       y = "Whether the Severity was Fatal (1) or Not (0)") + 
  ggtitle("The Severity of a Plane Accident as a Function of the Plane's 
          Safety Score") + 
  theme_bw()
#Note: The position_jitter parameter makes sure that not as many points overlap
#since there are a lot of points that take on the value of either 1 or 0. In
#essence, points that overlap with each other are simply placed above or below
#each other.
#Note: The following source was used to determine how to make points overlap
#with each other less:
#https://r-graphics.org/recipe-scatter-overplot
#Based on the results from the scatterplot, there is no evidence of complete
#separation, since there there are many safety scores for airplanes which can
#simultaneously result in a Severity that did or did not have fatalities.

#Now, let's plot a scatter plot of Days_Since_Inspection vs. 
#Severity_Binary_2
ggplot(data = aa3, aes(x = Days_Since_Inspection, y = Severity_Binary_2)) + 
  geom_point(alpha = 0.4, size = 1, position = position_jitter(width = 10, height = 0.1)) + 
  labs(x = "The number of days the plane was not inspected till the 
       accident occurred", 
       y = "Whether the Severity was Fatal (1) or Not (0)") + 
  ggtitle("The Severity of a Plane Accident as a Function of the Days Since 
          the Plane was Last Inspected Leading up to the Accident") + 
  theme_bw()
#Based on the results from the scatterplot, there is no evidence of complete
#separation, since there is an overlap for planes with the same x values
#that either have fatal or non-fatal accidents.

#Now, let's plot a scatter plot of Control_Metric vs. 
#Severity_Binary_2
ggplot(data = aa3, aes(x = Control_Metric, y = Severity_Binary_2)) + 
  geom_point(alpha = 0.4, size = 1, position = position_jitter(width = 10, height = 0.1)) + 
  labs(x = "The Plane's Control Metric Representing How Much Control the Pilot
       had During the Start of the Accident", 
       y = "Whether the Severity was Fatal (1) or Not (0)") + 
  ggtitle("The Severity of a Plane Accident as a Function of the Plane's 
          Control Metric") + 
  theme_bw()
#Based on the results from the scatterplot, there is no evidence of complete
#separation, since there is an overlap for planes with the same x values
#that either have fatal or non-fatal accidents.

#Thus far, there is no evidence at all for complete separation for all of the
#explanatory variables from m3.

#However, to further verify there is no complete separation, let's look at 
#the individual explanatory variables to determine whether they have large 
#standard errors and p-values
summary(m3)
#Based on the results, none of the explanatory variables have standard errors
#of greater than 5, which is good. Additionally, the p-value for
#each explanatory variable is less than alpha at 0.05. 
#Hereby, it would be
#reasonable to conclude that there is no evidence of complete separation.

### INTERPRETATIONS FOR THE FINAL GLM (Saul) ----

#In order to create an appropriate custom odds ratio, let's look at the
#smallest and greatest values each explanatory variable can take.

#Safety_Score:
min(aa3$Safety_Score, na.rm = TRUE)
#The smallest Safety_Score is 0.
max(aa3$Safety_Score, na.rm = TRUE)
#The largest Safety_Score is 100.
#Therefore, it would be reasonable to have a custom odds ratio for every
#10 unit increase in Safety_Score.

#Days_Since_Inspection:
min(aa3$Days_Since_Inspection, na.rm = TRUE)
#The smallest value for Days_Since_Inspection is 1.
max(aa3$Days_Since_Inspection, na.rm = TRUE)
#The largest value for Days_Since_Inspection is 23.
#Therefore, it would be reasonable to have a custom odds ratio for every
#1 unit increase in Safety_Score.

#Control_Metric:
min(aa3$Control_Metric, na.rm = TRUE)
#The smallest Control_Metric is around 21.
max(aa3$Control_Metric, na.rm = TRUE)
#The largest Control_Metric is around 98.
#Therefore, it would be reasonable to have a custom odds ratio for every
#5 unit increase in Safety_Score.

#Now, let's utilize the custom odds ratios determined to be reasonable
#from above to be used for the interpretations.
summary(m3)

#Safety_Score:
#For each 10 unit increase in the plane's safety score (as the rating increases, 
#the plane is deemed to be more safe for flight), the odds of a plane
#accident being fatal changes by a factor of e^(10*-0.058635), or rather decreases
#by a factor of around 0.556.

#Days_Since_Inspection:
#For each 1 unit increase in the number of days since
#the plane was last inspected, the odds of a plane
#accident being fatal changes by a factor of e^-0.198699, or rather decreases
#by a factor of around 0.82.

#Control_Metric:
#For each 5 unit increase in the plane's control metric (as the metric 
#increases, the pilot has more control over the plane at the start of the
#accident),
#the odds of a plane
#accident being fatal changes by a factor of e^(5*-0.014547), or rather decreases
#by a factor of around 0.930.

#95% Confidence Intervals:
confint(m3)
#The interpretation pertaining to Days_Since_Inspection and Severity_Binary_2
#was particularly interesting. Ironically, the more days pass since the plane
#was last inspected, the likelihood of that plane being in an accident decreases.
#It seems as though more frequent inspections would mean the plane is ensured
#to be in better quality and preparation for flight, but the custom odds
#interpretation proved otherwise.

#To make sure this actually holds true, let's look at the 95% confidence
#interval corresponding to this explanatory variable:

#We are 95% confident that the odds of a plane accident being fatal changes by
#a factor between around e^-0.233 and e^-0.165 for each 1 unit increase
#in the number of days since
#the plane was last inspected. Therefore, the custom odds interpretation does
#indeed hold true for this variable, since the confidence interval only displays
#values less than 1 for the e^Beta estimate for Days_Since_Inspection in relation
#to Severity_Binary_2. A value less than 1 for e^Beta represents a decrease.
#Hence, this verifies that for an increase in the number of days since
#the plane was last inspected, the likelihood of a plane being in a fatal
#accident decreases.

### LASSO REGRESSION (Nathan) ----

#Let's make fresh training and testing data with our new 1/0 response variable
RNGkind(sample.kind = "default")
set.seed(2291352)
train.idx <- sample(x = 1:nrow(aa3), size = .8*nrow(aa3))
train.df <- aa3[train.idx,]
test.df <- aa3[-train.idx,]

#Step 1 fit a traditional logistic regression to our training data
#We want to remove the factor variable severity_binary and only use Severity_binary_2 
#which is 1/0
train.df <- train.df %>% subset(select = -c(Severity_Binary))

#This will create a basic glm model using our new binary response variable
lr_mle <- glm(Severity_Binary_2 ~ . ,
              data = train.df, family = binomial(link = "logit"))

summary(lr_mle)
#No signs of complete separation because all the standard errors are very small
#with none of them even being above 1.

#save coefficients of the variables, to look at later
#Use pipelining
lr_mle_coefs <- lr_mle %>% coef()

#build x matrix (this "one-hot" codes everything), leaves the Y out.
x <- model.matrix(Severity_Binary_2 ~ ., data = train.df)

#create y as a vector object
y <- as.vector(train.df$Severity_Binary_2)

#alpha = 1 (default) fits lasso regression
#alpha = 0 fits ridge regression

#Start our lasso regression model using the x and y's we've established in the above code
lr_lasso <- glmnet(x=x, y=y, family = binomial(link = logit), alpha = 1)
summary(lr_lasso)

#Plot the regression so we get an idea how lambda is affecting
#our variable coefficents
plot(lr_lasso,xvar="lambda")

#Our goal is to find the optimal lambda that decreases the penalty
#on our coefficents

#See:
lr_lasso$lambda[1] #first lambda tried
## 0.1023237

#for that value of lambda, here is the corresponding beta vector
lr_lasso$beta[,1]
## all 0's...

lr_lasso$lambda[50] #50th value of lambda tried
## 0.00107196, so much smaller

#for that value of lambda, here is the corresponding beta vector
lr_lasso$beta[,50]
## All so tiny we can basically call them zero

#Lets start tuning using cross validation
lr_lasso_cv = cv.glmnet(x,y, family = binomial(link = "logit"))

plot(lr_lasso_cv)
#left hand side of the plot: MSE with less penalization (towards MLE estimated model) = -6.5
#right hand side of the plot: MSE with more penalization (towards an intercept only model) = little over -4
#We observe that we have a simple curve upwards as our lambda increases

#Pull our lambda min
lr_lasso_cv$lambda.min
#0.001417067

#Pull our lambda 1se
lr_lasso_cv$lambda.1se
#0.0191736

lr_lasso_coefs<- coef(lr_lasso_cv, s="lambda.1se") %>% as.matrix()
lr_lasso_coefs
#the optimal solution for out of sample prediction sets most coefficients exactly to 0

#create our x.test variable with only our predictors, so remove our Severity_Binary and explanatory variables
x.test <- model.matrix(Severity_Binary_2 ~ . -Severity_Binary, data = test.df)

#mutate test.df to add mle_pred and lasso_pred so we can plot roc curves
test.df <- test.df %>%
  
  mutate(mle_pred = predict(lr_mle, test.df, type = "response"),
         lasso_pred = predict(lr_lasso_cv, x.test, s = "lambda.1se", type = "response")[,1])

#Use this to get our estimated prediction values
predict(lr_lasso_cv, x.test, s = "lambda.1se", type = "response")[,1]

cor(test.df$mle_pred, test.df$lasso_pred)
#0.9613952

#Plot to witness the variability between our mle and lasso
test.df %>%
  ggplot() +
  geom_point(aes(x = mle_pred, y = lasso_pred)) +
  geom_abline(aes(intercept = 0, slope = 1))
#We observe a fairly consistent variability meaning that there isn't a noticeable difference between the two models.

library(pROC) #Just to double check we have the correct package

#see if ROC supports this
#create ROC curves
str(test.df)

#Create an ROC curve for mle so we can establish a baseline to compare to our lasso model
mle_rocCurve   <- roc(response = test.df$Severity_Binary,#supply truth
                      predictor = test.df$mle_pred,#supply predicted PROBABILITIES
                      levels = c("No", "Yes") #(negative, positive)
)

str(test.df)

#plot basic ROC curve with mle model
plot(mle_rocCurve, print.thres = TRUE,print.auc = TRUE)
#AUC: 0.766

#Not a bad AUC but lets see if lasso is any better
lasso_rocCurve   <- roc(response = test.df$Severity_Binary,#supply truth
                        predictor = test.df$lasso_pred,#supply predicted PROBABILITIES
                        levels = c("No", "Yes") #(negative, positive)
)

#plot basic ROC curve with Lasso model
plot(lasso_rocCurve, print.thres = TRUE,print.auc = TRUE)
#AUC: 0.776

#The AUC from lasso is better! not by a ton but it's a enough to bring lasso into consideration
str(lasso_rocCurve)
ggplot() +
  geom_line(aes(x = 1-mle_rocCurve$specificities, y = mle_rocCurve$sensitivities), colour = "darkorange1") +
  geom_text(aes(x = .75, y = .75,
                label = paste0("MLE AUC = ",round(mle_rocCurve$auc, 3))), colour = "darkorange1")+
  geom_line(aes(x = 1-lasso_rocCurve$specificities, y = lasso_rocCurve$sensitivities), colour = "cornflowerblue")+
  geom_text(aes(x = .75, y = .65,
                label = paste0("Lasso AUC = ",round(lasso_rocCurve$auc, 3))), colour = "cornflowerblue") +
  labs(x = "1-Specificity", y = "Sensitivity")
#So looking at this graph we can see how the MLE and the Lasso model compete with each other. We can observe
#that the Lasso model holds a lead over the MLE model by roughly .01

# VISUALIZATIONS (Hunter) ----

#Since all three of our explanatory variables are numeric or integers, histograms
#will be used to explore the visualizations between themselves and the binary y
#y variable. To do this, we will use ggplot's histogram feature. We will set the 
#x variable to the explanatory variable we are looking to explore and fill with the
#response variable (Severity_Binary). From there, we will add labels and change
#the color of the graph so that it is color-blind friendly. 

ggplot(data = aa3) +
  geom_histogram(aes(x = Safety_Score, fill = Severity_Binary), position="fill", binwidth = 5) +
  labs(x = "Safety Score", y = "Proportion") + #Setting x and y labels 
  scale_fill_grey("Crash \nSevere") + #Setting the legend for the graph
  ggtitle("Crash Severity by Safety Score of Plane") #Setting the title for the graph
#The histogram above reveals that there is a higher proportion of fatal plane crashes
#when safety score gets closer to either 0 or 100. This can be determined by the increased
#proportion of non-fatal plane crashes as the safety score nears 50. 

ggplot(data = aa3) +
  geom_histogram(aes(x = Days_Since_Inspection, fill = Severity_Binary), position = "fill", binwidth = 2) +
  labs(x = "Days Since Inspection", y = "Proportion") +
  scale_fill_grey("Crash \nSevere") +
  ggtitle("Crash Severity by Days Since Last Inspection of Plane")
#Based on the histogram between the explanatory variable days since inspection and
#the response variable crash severity, a great portion of fatal crashes occur when
#the last inspection was more than 2 days ago. This can be seen in the graph as the first
#bin has the highest proportion of non-fatal crashes. After two days,
#the proportion of fatal crashes increases to roughly .60 and then begins to drop.
#An interesting observation is that the second lowest proportion of fatal crashes
#occurred in the last bin from 21-23 days. Intuition would tell us that as the days
#since last inspection increases, the proportion of fatal crashes would increase.
#This is not the case though. 

ggplot(data = aa3) +
  geom_histogram(aes(x = Control_Metric, fill = Severity_Binary), position = "fill", binwidth = 7) +
  labs(x = "Control Metric", y = "Proportion") +
  scale_fill_grey("Crash \nSevere") +
  ggtitle("Crash Severity by Control Metric")
#Based on the explanatory variable control metric, you would think that there would
#be a great proportion of fatal crashes as the control metric decreases, since the 
#pilot was in less control of the situation. The histogram paints a slightly different
#picture. The proportion of fatal crashes does decrease as the control metric decreases
#but only until the variable hits 25, where the proportion of fatal crashes dramatically
#increases.

# MULTIVARIATE VISUALIZATIONS (Hunter) ------

#For our multivariate visualizations, we will take two of the three variables that were
#important and plot them together to see if any patterns exist. Since all three
#of our important variables are numeric, we will use scatter plots for all graphs
ggplot(data = aa3) +
  geom_point(aes(x = Safety_Score, y = Control_Metric, colour = Severity_Binary)) +
  labs(y = "Control Metric", x = "Safety Score") +
  scale_colour_brewer("Crash\nSevere",palette = "Paired") +
  ggtitle("Crash Severity by Control Metric and Safety Score")
#No patterns arise based on the scatter plot between Safety Score and control metric.
#The plot looks randomly scattered with no pattern.

ggplot(data = aa3) +
  geom_point(aes(x = Safety_Score, y = Days_Since_Inspection, colour = Severity_Binary)) +
  labs(y = "Days Since Insepction", x = "Safety Score") +
  scale_colour_brewer("Crash\nSevere",palette = "Paired") +
  ggtitle("Crash Severity by Days Since Inspection and Safety Score")
#The scatter plot between Safety score and days since inspection
#reveals an interesting pattern. For each day since inspection, there
#are four distinct safety score ranges. The category with the lowest
#safety score has almost all fatal crashes. The middle two
#categories have mostly non-fatal crashes and the final category
#has more fatal crashes than non-fatal. Additionally, as the days
#since inspection decrease, the safety score categories increase
#but the trend within the four categories remains the same.

#Let's create histograms separated by days since inspection to
#see if we can paint a better picture
ggplot(data = aa3) +
  geom_histogram(aes(x = Safety_Score, fill = Severity_Binary), binwidth = 7, position = "fill") +
  facet_wrap(~I(Days_Since_Inspection > 14)) + #splits the days that are under 14 days (False) and over 14 days (True)
  labs(x = "Safety Score", y = "Count") +
  scale_fill_brewer("Crash\nSevere",palette ="Paired") +
  ggtitle("Crash Severity by Days Since Inspection and Safety Score")

ggplot(data = aa3) +
  geom_histogram(aes(x = Safety_Score, fill = Severity_Binary), binwidth = 7) +
  facet_wrap(~I(Days_Since_Inspection > 14)) + #splits the days that are under 14 days (False) and over 14 days (True)
  labs(x = "Safety Score", y = "Count") +
  scale_fill_brewer("Crash\nSevere",palette ="Paired") +
  ggtitle("Crash Severity by Days Since Inspection and Safety Score")
#The multivariate histogram reveals more about the number of observations under
#each day since inspection. Overall, there are more likely to be fatal crashes for planes that
#were inspected less than 14 days. Yet, the likelihood of a fatal crash still
#more or less remains the same for days less than or more than 14.

#Therefore, this tells the client that for the most part, regardless of how many days pass since
#a plane was last inspected, that plane's safety score is ultimately crucial
#for ensuring a plane is less likely to get into an accident.

ggplot(data = aa3) +
  geom_point(aes(x = Control_Metric, y = Days_Since_Inspection, colour = Severity_Binary)) +
  labs(y = "Days Since Insepction", x = "Control Metric") +
  scale_colour_brewer("Crash\nSevere",palette = "Paired") +
  ggtitle("Crash Severity by Days Since Inspection and Control Metric")
#The scatter plot between control metric and days since inspection also reveals
#no patterns or trends.

#MORE VISUALIZATIONS (Saul) ------
ggplot(data = aa3) +
  geom_histogram(aes(x = Safety_Score, fill = Severity_Binary), binwidth = 5) +
  labs(x = "Safety Score", y = "Proportion") + #Setting x and y labels 
  scale_fill_grey("Crash \nSevere") + #Setting the legend for the graph
  ggtitle("Crash Severity by Safety Score of Plane") #Setting the title for the graph
#Although it is interesting to note that an increase in a plane's safety score
#(after 50) results in an increased likelihood of a fatal accident (refer to
#above ggplot histogram for Safety_Score vs. Severity_Binary under VISUALIZATIONS), this graph
#portrays that having a plane with a very large safety score be in an 
#accident is
#actually a very rare occurrence. We can see that this is a very rare occurrence,
#because the bars indicating the crash Severity of "Yes" and "No"
#are very small when the Safety_Score is close to 100. In fact, as the
#Safety_Score increases and gets closer to 100, the bars keep getting smaller
#and smaller. Hence, this demonstrates that overall any kind of plane accident
#for a plane with a high Safety_Score occurs rarely.

#When looking at the safety scores, most planes have safety scores between
#25 and 75, as the bars there are taller. When looking specifically 
#at safety scores
#between 25 and 75,
#the higher the Safety_Score, the less likelihood the plane will have a fatal
#accident; the lower the Safety_Score, the greater likelihood the plane
#will have a fatal accident.

#Ultimately, this tells our client that a plane's safety score is very
#important to consider as a factor for affecting a plane's accident, yet it's
#also important to be aware that simply because a plane has a high safety score
#doesn't guarantee that it won't be in a fatal accident.

ggplot(data = aa3) +
  geom_histogram(aes(x = Days_Since_Inspection, fill = Severity_Binary), binwidth = 2) +
  labs(x = "Days Since Inspection", y = "Proportion") +
  scale_fill_grey("Crash \nSevere") +
  ggtitle("Crash Severity by Days Since Last Inspection of Plane")
#It's indeed interesting that the more days
#that pass since a plane was last inspected, the plane is less likely to
#be in a fatal accident (refer to the above ggplot histogram
#for Days_Since_Inspection vs. Severity_Binary under VISUALIZATIONS). 
#Yet, similar to the above graph, it is rare for
#a plane to not be inspected for more than 20 days, as indicated by small bars
#for the severity of the crash. However, when looking at the graph, we see
#that most planes were last inspected between 5 and 20 days before takeoff.
#Looking at those values, it displays interesting findings.
#When the days since the plane was last inspected goes from 5 to around 12 days,
#the likelihood of a plane crash being fatal increases. Yet, after the plane
#goes past 15 days of being less inspected, the likelihood of a plane crash
#being fatal begins to decrease. Although an assumption, this probably
#tells the client that if a plane was not last inspected between 15-20 days, 
#it's
#probably a sign that the plane didn't really need to be inspected as it may
#have been in very good shape without having to have frequent inspections.

ggplot(data = aa3) +
  geom_histogram(aes(x = Control_Metric, fill = Severity_Binary), binwidth = 7) +
  labs(x = "Control Metric", y = "Proportion") +
  scale_fill_grey("Crash \nSevere") +
  ggtitle("Crash Severity by Control Metric")
#Ironically, as noted in the ggplot histogram for Control_Metric vs.
#Severity_Binary under VISUALIZATIONS, the proportion of fatal crashes increases as the control metric
#becomes greater than 25. At some point on the graph, an increase in the
#Control_Metric then starts to result in a decreased likelihood of fatal
#crashes. However, similar to the above graphs, as the control metric becomes
#closer to 100, it becomes rarer for planes to be in any kind of accident.
#This may tell the client that it's important for the pilot to be in
#control of their situation and flight, yet having too much control might
#mean that the pilot is pretty anxious and gets too fixated on small details
#and overlooks the bigger picture; this may increase the likelihood of
#a fatal accident.