library(caret)
library(xgboost)
library(data.table)
dat <- fread("XXX:/....csv")
setnames(dat, old=names(dat), new=gsub("\\.", "_", names(dat)))
dat[, Lon2 := Lon_^2][, Lat2 := Lat_^2][, LonLat := Lon_ * Lat_]
y <- dat$ESV_Sum
X <- dat[, .(SLR, GDP, population, Human_footprint, sedimentary, Storme, PA, Lon_, Lat_, Lon2, Lat2, LonLat)]
grid <- expand.grid(
  nrounds = c(150, 200),
  eta = c(0.05, 0.1),
  max_depth = c(3, 4, 5),
  gamma = c(0, 1),
  colsample_bytree = c(0.7, 0.9),
  min_child_weight = c(1, 3),
  subsample = c(0.7, 0.9)
)
ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE, allowParallel = TRUE)
set.seed(123)
fit <- train(
  x = X, y = y, 
  method = "xgbTree", 
  trControl = ctrl, 
  tuneGrid = grid,
  metric = "RMSE"
)
cat("Best params:\n")
print(fit$bestTune)
saveRDS(fit, "XXX:/....rds")
