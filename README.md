# Global-patterns-and-drivers-of-tidal-wetland-ecosystem-service-value-dynamics-1999-2019-
Related code and data attachments of the article

This repository contains code and supporting workflows for reproducing the analysis of tidal wetland ecosystem service value (ESV) dynamics from 1999 to 2019. The pipeline covers regional raster clipping, point-based value preparation, neighborhood-based gap filling, scaling-factor integration, and yearly gain/loss accounting.

- `1.Download_tifdata.ipynb`: data acquisition / preparation notebook
- `2.Cliptif_by_region.ipynb`: regional GeoTIFF clipping
- `3.Clip_origin_data.py`: split source point records into region-specific CSV files
- `4.Calculate_value.py`: raster-to-point conversion and 5/10/50 km + median filling workflow
- `5.S1S2.py`: merge S1/S2 scaling factors into filled CSV outputs
- `6.Net_gain_with_weight.py`: compute yearly delta, cumulative values, and final totals
- `7.demo.py`: one-run demonstration pipeline for a single example case

-“Temporal Trend Analysis of ESV change.R” fits a quadratic regression model to time series data and generates predictions with both confidence and prediction intervals.

-“XGBoost.R” performs hyperparameter tuning and cross-validated training of an XGBoost regression model on spatial and environmental predictor data using the caret framework.

-“SHAP.R” calculates SHAP values for a trained XGBoost model to interpret the importance and contribution of each predictor variable in the model’s predictions. This R script exports both the global feature importances and the individual sample-level SHAP values for further analysis.

-"FigureData.xlsx" file is used to store the data required for drawing figures in the paper.

### Notes
Some processing scripts depend on ArcPy and therefore require a licensed local ArcGIS Pro environment. These scripts are included for transparency and methodological reproducibility.
