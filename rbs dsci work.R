rbs_dsci <- read.csv("/Users/jonahlubin/Desktop/rbs_data_updated_dsci303.csv")
rbs_dsci <- rbs_dsci[,-1]

rbs_dsci_num <- rbs_dsci[,-1]

library(corrplot)
library(ggplot2)
library(grid)
library(MASS)
library(dplyr)

rbs_dsci_trimmed <- rbs_dsci[,c(2:29, 42)]
rbs_cor_trimmed <- cor(rbs_dsci_trimmed, use = "pairwise.complete.obs")

pdf(file = "rbs_cor_matrix.pdf")
corrplot(rbs_cor_trimmed, type="lower", tl.cex=.55, tl.col="black", diag = FALSE, method = "color")
dev.off()

rbs_dsci_trimmed_2 <- rbs_dsci[,c(1:29, 42)]

colSums(is.na(rbs_dsci_trimmed_2))

rbs_dsci_trimmed_2[is.na(rbs_dsci_trimmed_2)] <- 50

colSums(is.na(rbs_dsci_trimmed_2))

rbs.lm<- lm(nfl_grades_offense ~ college_player_game_count + college_receptions + college_breakaway_yards_per_att + college_grades_hands_fumble + college_grades_offense + college_avoided_tackles_per_touch + college_elu_rush_mtf_per_att + college_elu_yco_per_touch + college_fumbles_per_touch + college_run_to_pass_plays_ratio + college_catch_percentage, data = rbs_dsci_trimmed_2)
summary(rbs.lm)

step_model <- stepAIC(rbs.lm, direction = "both", 
                      trace = FALSE)

summary(step_model)

par(mfrow=c(2,2))
plot(step_model)
par(mfrow=c(1,1))

#############################

clustering_subset_rbs <- rbs_dsci[,c(1:29)]

clustering_subset_rbs[is.na(clustering_subset_rbs)] <- 50
clustering_subset_rbs_2 <- clustering_subset_rbs[,-1]
scaled_data <- scale(clustering_subset_rbs_2)

set.seed(123) 
k <- 3
kmeans_result <- kmeans(scaled_data, centers = k)

cluster_assignments <- kmeans_result$cluster

clustering_subset_rbs$cluster_assignments <- cluster_assignments

names(clustering_subset_rbs)
player_archetype <- clustering_subset_rbs %>%
  select(college_player, cluster_assignments)

table(clustering_subset_rbs$cluster_assignments)

pca_result <- prcomp(clustering_subset_rbs_2, scale. = TRUE)

# Extract the first two principal components
PC1 <- pca_result$x[, 1]
PC2 <- pca_result$x[, 2]

# Combine with cluster assignments
pc_data <- data.frame(PC1, PC2, cluster_assignments = as.factor(cluster_assignments), college_player = clustering_subset_rbs$college_player)

# Create a scatter plot
library(ggplot2)
library(ggrepel)

clustering_dsci_plot_k3 <- ggplot(pc_data, aes(x = PC1, y = PC2, color = cluster_assignments)) +
  geom_point() +
  labs(title = "RB Clustering")

ggsave("clustering_dsci_plot_k3.png", clustering_dsci_plot_k3)

############################

library(randomForest)
rbs_dsci_trimmed <- rbs_dsci[,c(2:29, 42)]
rbs_dsci_trimmed[is.na(rbs_dsci_trimmed)] <- 50

set.seed(123)
train_indices <- sample(1:nrow(rbs_dsci_trimmed), 0.8 * nrow(rbs_dsci_trimmed))  # 80% for training
train_data <- rbs_dsci_trimmed[train_indices, ]
test_data <- rbs_dsci_trimmed[-train_indices, ]

rf_model <- randomForest(nfl_grades_offense ~ ., data = train_data, ntree=500)
print(rf_model)

predictions <- predict(rf_model, newdata = test_data)

accuracy <- sqrt(mean((predictions - test_data$nfl_grades_offense)^2))
print(paste("Root Mean Squared Error on Test Set: ", round(accuracy, 2)))

library(caret)
rsquared_value <- R2(test_data$nfl_grades_offense, predictions)
print(rsquared_value)

predictions <- as.data.frame(predictions)
test_data <- cbind(test_data, predictions)

random_forest_rbs_dsci <- ggplot(data = test_data, aes(x = nfl_grades_offense, y = predictions)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Actual vs. Predicted NFL Grades on Test Data Using Random Forests", x = "Actual NFL Grade", y = "Predicted NFL Grade")

ggsave("random_forest_rbs_dsci.png", random_forest_rbs_dsci)



varImpPlot(rf_model)

options(repos='http://cran.rstudio.org')
have.packages <- installed.packages()
cran.packages <- c('devtools','plotrix','randomForest','tree')
to.install <- setdiff(cran.packages, have.packages[,1])
if(length(to.install)>0) install.packages(to.install)

library(devtools)
if(!('reprtree' %in% installed.packages())){
  install_github('munoztd0/reprtree')
}
for(p in c(cran.packages, 'reprtree')) eval(substitute(library(pkg), list(pkg=p)))

library(randomForest)
library(reprtree)

model <- randomForest(nfl_grades_offense ~ ., data=train_data, importance=TRUE, ntree=300, mtry = 2, do.trace=10)
print(model)
pdf(file = "tree_plot.pdf", width=40,height=10)
reprtree:::plot.getTree(model)
dev.off()

################################

library(xgboost)
rbs_dsci_trimmed <- rbs_dsci[,c(2:29, 42)]
rbs_dsci_trimmed[is.na(rbs_dsci_trimmed)] <- 50

rbs_dsci_trimmed <- model.matrix(~ . - 1 + nfl_grades_offense, data = rbs_dsci_trimmed)

set.seed(123)
train_index <- sample(seq_len(nrow(rbs_dsci_trimmed)), size = 0.8 * nrow(rbs_dsci_trimmed))
train_data <- rbs_dsci_trimmed[train_index, ]
test_data <- rbs_dsci_trimmed[-train_index, ]

features <- colnames(rbs_dsci_trimmed)[-which(colnames(rbs_dsci_trimmed) == "nfl_grades_offense")]
response <- "nfl_grades_offense"

xgb_model <- xgboost(
  data = as.matrix(train_data[, features]),
  label = train_data[, response],
  nrounds = 100,  # Number of boosting rounds
  objective = "reg:squarederror",  # Specify the objective for regression task
  max_depth = 3,  # Maximum depth of each tree
  eta = 0.3  # Learning rate
)

predictions <- predict(xgb_model, as.matrix(test_data[, features]))

rmse <- sqrt(mean((test_data[, response] - predictions)^2))
print(paste("Root Mean Squared Error:", rmse))

test_data <- as.data.frame(test_data)

str(test_data)

predictions <- as.data.frame(predictions)
test_data <- cbind(test_data, predictions)

rsquared_value <- R2(test_data$nfl_grades_offense, test_data$predictions)
print(rsquared_value)

xgboost_rbs_dsci <- ggplot(data = test_data, aes(x = nfl_grades_offense, y = predictions)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(title = "Actual vs. Predicted NFL Grades on Test Data Using Gradient Boosting", x = "Actual NFL Grade", y = "Predicted NFL Grade")

ggsave("xgboost_rbs_dsci.png", xgboost_rbs_dsci)

