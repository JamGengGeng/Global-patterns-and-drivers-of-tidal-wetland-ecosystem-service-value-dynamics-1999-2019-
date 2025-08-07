library(ggplot2)
data <- data.frame(
  year = c(2001, 2004, 2007, 2010, 2013, 2016, 2019),
  value = c(-23.9628, -23.9628, -26.6357, -20.1057, -20.7543, -35.8907, -39.2928)
)
data$y2 <- data$year^2
fit <- lm(value ~ year + y2, data)
summary(fit)
cat("R2=", signif(summary(fit)$r.squared, 3), "; P=", signif(pf(summary(fit)$fstat[1], summary(fit)$fstat[2], summary(fit)$fstat[3], lower.tail=FALSE), 3), "\n")
yr_pred <- 2001:2030
pred_df <- data.frame(year=yr_pred, y2=yr_pred^2)
pint <- predict(fit, pred_df, interval="prediction")
cint <- predict(fit, pred_df, interval="confidence")
pred_df <- cbind(pred_df, fit=pint[, "fit"], lwr=pint[, "lwr"], upr=pint[, "upr"], clwr=cint[, "lwr"], cupr=cint[, "upr"])
ggplot(pred_df, aes(year, fit)) +
  geom_ribbon(aes(ymin=lwr, ymax=upr), fill="grey80", alpha=0.5) +
  geom_ribbon(aes(ymin=clwr, ymax=cupr), fill="grey60", alpha=0.4) +
  geom_line(color="blue") +
  geom_point(data=data, aes(year, value), color="black", size=2) +
  theme_bw()
