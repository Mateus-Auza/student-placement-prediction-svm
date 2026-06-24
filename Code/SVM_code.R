# Libraries
library(LiblineaR)
library(skimr)
library(ggcorrplot)
library(ggplot2)
library(kernlab)
library(doParallel)
library(parallel)
library(caret)
library(patchwork)

# -------------------------
# Load and clean data
# -------------------------
stud <- read.csv("student_placement_prediction_dataset_2026.csv")

stud$student_id <- NULL
stud$salary_package_lpa <- NULL
stud$placement_status <- as.factor(stud$placement_status)

skim(stud)

# -------------------------
# Preprocessing
# -------------------------
y <- stud$placement_status

X <- model.matrix(~ . - placement_status, data = stud)[, -1]
X <- as.data.frame(X)

set.seed(123)
train_index <- sample(1:nrow(X), 0.8 * nrow(X))

X_train <- X[train_index, ]
X_test  <- X[-train_index, ]

y_train <- y[train_index]
y_test  <- y[-train_index]

X_train <- scale(X_train)
X_test  <- scale(
  X_test,
  center = attr(X_train, "scaled:center"),
  scale  = attr(X_train, "scaled:scale")
)

train_data <- data.frame(X_train, placement_status = y_train)
test_data  <- data.frame(X_test,  placement_status = y_test)

# -------------------------
# EDA
# -------------------------
ggplot(stud, aes(x = placement_status)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Placement Status Distribution",
       x = "Placement Status",
       y = "Count")

cor_matrix <- cor(X_train)

ggcorrplot(cor_matrix, lab = FALSE) +
  labs(title = "Correlation Matrix")

# -------------------------
# PCA
# -------------------------
pca_model <- prcomp(X_train, center = TRUE, scale. = TRUE)

explained_var <- pca_model$sdev^2
explained_var_ratio <- explained_var / sum(explained_var)

scree_data <- data.frame(
  PC = 1:length(explained_var_ratio),
  Variance = explained_var_ratio
)

p1 <- ggplot(scree_data, aes(x = PC, y = Variance)) +
  geom_line() +
  geom_point() +
  labs(
    title = "Scree Plot",
    x = "Principal Component",
    y = "Proportion of Variance Explained"
  ) +
  theme_minimal()

p1

pca_data1 <- data.frame(
  PC1 = pca_model$x[, 1],
  PC2 = pca_model$x[, 2],
  placement_status = y_train
)

p2 <- ggplot(pca_data1, aes(x = PC1, y = PC2, color = placement_status)) +
  geom_point(alpha = 0.7) +
  labs(
    x = paste0("PC1 (", round(100 * explained_var_ratio[1], 2), "%)"),
    y = paste0("PC2 (", round(100 * explained_var_ratio[2], 2), "%)")
  ) +
  theme_minimal()

pca_data2 <- data.frame(
  PC3 = pca_model$x[, 3],
  PC4 = pca_model$x[, 4],
  placement_status = y_train
)

p3 <- ggplot(pca_data2, aes(x = PC3, y = PC4, color = placement_status)) +
  geom_point(alpha = 0.7) +
  labs(
    x = paste0("PC3 (", round(100 * explained_var_ratio[3], 2), "%)"),
    y = paste0("PC4 (", round(100 * explained_var_ratio[4], 2), "%)")
  ) +
  theme_minimal()

(p2 | p3) + patchwork::plot_layout(guides = "collect")

# -------------------------
# Linear SVM
# -------------------------
linear_svm <- LiblineaR(
  data = as.matrix(X_train),
  target = y_train,
  type = 1,
  cost = 1
)

pred_linear <- predict(linear_svm, as.matrix(X_test))$predictions
pred_linear <- as.factor(pred_linear)

w_full <- as.vector(linear_svm$W)
w_vec <- w_full[1:28]
b <- w_full[29]

X1 <- as.matrix(X_train)
y1 <- ifelse(y_train == levels(y_train)[1], -1, 1)

margins <- y1 * (X1 %*% w_vec + b)

support_vectors <- margins <= 1
num_sv <- sum(support_vectors)

hinge_loss <- pmax(0, 1 - margins)

C <- 1
objective_value <- 0.5 * sum(w_vec^2) + C * sum(hinge_loss)

results_df <- data.frame(
  num_support_vectors = num_sv,
  objective_function_value = objective_value
)

results_df

caret::confusionMatrix(pred_linear, y_test)

# -------------------------
# RBF SVM (caret)
# -------------------------
cores <- detectCores() - 1
cl <- makeCluster(cores)
registerDoParallel(cl)

set.seed(123)
small_train <- train_data[sample(1:nrow(train_data), 1000), ]

ctrl <- trainControl(method = "cv", number = 5, allowParallel = TRUE)

svm_model <- train(
  placement_status ~ .,
  data = small_train,
  method = "svmRadial",
  tuneGrid = expand.grid(
    C = c(0.1, 1, 10),
    sigma = c(0.01, 0.1, 1)
  ),
  trControl = ctrl
)

stopCluster(cl)

results <- svm_model$results
best <- svm_model$bestTune
best_point <- merge(results, best)

ggplot(results, aes(x = sigma, y = Accuracy, color = factor(C))) +
  geom_line() +
  geom_point() +
  geom_point(
    data = best_point,
    aes(x = sigma, y = Accuracy),
    color = "red",
    size = 4
  ) +
  geom_text(
    data = best_point,
    aes(label = "Best"),
    vjust = -1,
    color = "red"
  ) +
  labs(
    title = "SVM (RBF) Hyperparameter Tuning",
    x = "Gamma (sigma)",
    y = "Cross-Validated Accuracy",
    color = "Cost (C)"
  ) +
  theme_minimal()

pred_rbf <- predict(svm_model, test_data)

rbf_fit <- svm_model$finalModel

num_sv <- nrow(rbf_fit@xmatrix[[1]])

alpha <- as.vector(rbf_fit@coef[[1]])
SV <- rbf_fit@xmatrix[[1]]

K <- as.matrix(kernelMatrix(
  rbfdot(sigma = rbf_fit@kernelf@kpar$sigma),
  SV
))

objective_value <- sum(abs(alpha)) - 0.5 * t(alpha) %*% K %*% alpha

result <- data.frame(
  num_support_vectors = num_sv,
  objective_function_value = as.numeric(objective_value)
)

result

caret::confusionMatrix(pred_rbf, y_test)

# -------------------------
# Model Comparison + PCA plots
# -------------------------
Metric <- c(
  "Accuracy",
  "Balanced Accuracy",
  "Recall (Not Placed)",
  "Recall (Placed)",
  "F1-score (Not Placed)",
  "F1-score (Placed)",
  "Support Vectors (%)"
)

Linear_SVM <- c(0.576, 0.555, 0.331, 0.779, 0.41, 0.67, 0.80)
RBF_SVM    <- c(0.552, 0.525, 0.225, 0.824, 0.31, 0.67, 0.92)

model_comparison <- data.frame(
  Metric,
  Linear_SVM,
  RBF_SVM
)

model_comparison

pca_model <- prcomp(X_train, scale. = TRUE)

explained_var_ratio <- pca_model$sdev^2 / sum(pca_model$sdev^2)

X_train_pca <- predict(pca_model, X_train)
X_test_pca  <- predict(pca_model, X_test)

plot_data <- data.frame(
  PC1 = X_test_pca[,1],
  PC2 = X_test_pca[,2],
  true = y_test,
  pred = pred_linear
)

p7 <- ggplot(plot_data, aes(PC1, PC2, color = pred)) +
  geom_point(size = 2) +
  labs(
    title = "Linear SVM",
    x = paste0("PC1 (", round(100 * explained_var_ratio[1], 2), "%)"),
    y = paste0("PC2 (", round(100 * explained_var_ratio[2], 2), "%)")
  ) +
  theme_minimal()

plot_data_rbf <- data.frame(
  PC1 = X_test_pca[,1],
  PC2 = X_test_pca[,2],
  true = y_test,
  pred = pred_rbf
)

p8 <- ggplot(plot_data_rbf, aes(PC1, PC2, color = pred)) +
  geom_point(size = 2, alpha = 0.7) +
  labs(
    title = "RBF SVM",
    x = paste0("PC1 (", round(100 * explained_var_ratio[1], 2), "%)"),
    y = paste0("PC2 (", round(100 * explained_var_ratio[2], 2), "%)")
  ) +
  theme_minimal()

(p7 | p8) + patchwork::plot_layout(guides = "collect")
