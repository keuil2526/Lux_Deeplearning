| Column                        | Meaning                                         | Importance               |
| ----------------------------- | ----------------------------------------------- | ------------------------ |
| target_lux                    | Center Illuminance in lux (Label)               | ★★★★★                |
| C_mean_luma                   | Mean luma of center ROI                         | ★★★★★                |
| C_std_luma                    | Std luma of center ROI                          | ★★★★☆                |
| UL/UR/LR/LL_mean_luma         | Mean luma of each corner ROI                    | ★★★★☆                |
| corner_mean_luma              | Mean luma over 4 corner ROIs                    | ★★★★☆                |
| square_mean_luma              | Mean luma of full square region                 | ★★★★☆                |
| C_minus_corner_mean_luma      | Center - Corner luma (spatial contrast)         | ★★★★☆                |
| C_over_corner_mean_luma       | Center / Corner luma ratio                      | ★★★★☆                |
| square_mean_sat_proxy         | HSV saturation proxy of square region           | ★★★☆☆                |
| square_grid_range_mean_luma   | Range of 5x5 grid cell mean lumas               | ★★★☆☆                |
| square_grid_std_mean_luma     | Std of 5x5 grid cell mean lumas                 | ★★★☆☆                |
| grad_mean / grad_std          | Gradient features (texture/edge strength)       | ★★★☆☆                |
| square_bottom_minus_top_luma  | Vertical luma gradient                          | ★★★☆☆                |
| square_right_minus_left_luma  | Horizontal luma gradient                        | ★★☆☆☆                |
| lux_UL/UR/LR/LL               | Corner lux measurements (NaN for 1point)        | ★★★★★ (label only)   |
| session                       | Acquisition session ID                          | ★★★★☆ (grouping)     |
| surface_group_norm            | Surface category (white_paper/table/etc.)       | ★★★☆☆ (grouping)     |
| image_path / image_name       | File path info                                  | ★☆☆☆☆ (metadata)    |


## Notes
- target_lux == lux_C == central_lux : all the same value (center lux)
- lux_UL/UR/LR/LL : NaN for 1point images (only center measured)
- Used KNeighborsRegressor (regression task, not classification)
- Best K=1, R2=0.9197, MAE=52.72 lux on test set
