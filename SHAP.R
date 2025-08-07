library(treeshap)
library(data.table)
mod <- readRDS("XXX:/.....rds")
dat <- fread("XXX:/....csv")
names(dat) <- gsub("\\.", "_", names(dat))
dat[, Lon2 := Lon_^2][, Lat2 := Lat_^2][, LonLat := Lon_ * Lat_]
feats <- c("Sea_level", "GDP", "Population", "Human_footprint", "sediment_supply", "Storm surge", "Protected_area", "Lon_", "Lat_", "Lon2", "Lat2", "LonLat")
X <- dat[, ..feats]
Xmat <- model.matrix(~ . - 1, data = X)
tdf <- xgboost.unify(mod$finalModel, Xmat)
ts <- treeshap(tdf, Xmat)
shap_global <- colMeans(abs(ts$shaps))
shap_df <- data.frame(Feature = names(shap_global), Importance = as.numeric(shap_global))
shap_df <- shap_df[order(shap_df$Importance), ]
write.csv(shap_df, "XXX:/....csv", row.names = FALSE)
shap_val <- as.data.frame(ts$shaps)
shap_val$Sample_ID <- 1:nrow(shap_val)
if (all(c("Lon_", "Lat_") %in% colnames(X))) {
  shap_val$Lon <- X$Lon_
  shap_val$Lat <- X$Lat_
}
fwrite(shap_val, "XXX:/....csv")
