from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit as gss
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.impute import SimpleImputer as si
from sklearn.pipeline import make_pipeline as mp
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#출력 구간 구분
def printer(text):
    print()
    print('='*10, text, '='*10)
    print()

#사용할 특징 설정
feature_sets = {
    'Central': ['C_mean_luma', 'C_std_luma', 'C_grad_mean', 'C_grad_std'],
    'Five_Regions': [
        'C_mean_luma', 'C_std_luma', 'C_grad_mean', 'C_grad_std',
        'UL_mean_luma', 'UL_std_luma', 'UL_grad_mean', 'UL_grad_std',
        'UR_mean_luma', 'UR_std_luma', 'UR_grad_mean', 'UR_grad_std',
        'LR_mean_luma', 'LR_std_luma', 'LR_grad_mean', 'LR_grad_std',
        'LL_mean_luma', 'LL_std_luma', 'LL_grad_mean', 'LL_grad_std'
    ],
    'Brightness_Statistics': [
        'C_mean_luma', 'C_std_luma', 'corner_mean_luma',
        'corner_std_luma', 'corner_range_luma',
        'square_mean_luma', 'square_std_luma'
    ],
    'All_35_Features': [
            "C_mean_luma",
            "C_std_luma",
            "C_grad_mean",
            "C_grad_std",
            "UL_mean_luma",
            "UL_std_luma",
            "UL_grad_mean",
            "UL_grad_std",
            "UR_mean_luma",
            "UR_std_luma",
            "UR_grad_mean",
            "UR_grad_std",
            "LR_mean_luma",
            "LR_std_luma",
            "LR_grad_mean",
            "LR_grad_std",
            "LL_mean_luma",
            "LL_std_luma",
            "LL_grad_mean",
            "LL_grad_std",
            "corner_mean_luma",
            "corner_std_luma",
            "corner_range_luma",
            "C_minus_corner_mean_luma",
            "C_over_corner_mean_luma",
            "square_mean_luma",
            "square_std_luma",
            "square_grad_mean",
            "square_grad_std",
            "square_mean_sat_proxy",
            "square_bottom_minus_top_luma",
            "square_right_minus_left_luma",
            "square_center_minus_corners_luma",
            "square_grid_range_mean_luma",
            "square_grid_std_mean_luma"
        ]
}

#전처리 데이터 불러오기
df = pd.read_csv('data/lux_preprocessed.csv')
y = df['target_lux']
printer('Data Load Completed')
print('Data Size : ', df.shape)

#학습 데이터와 테스트 데이터 분리
if 'session' in df.columns:
    splitter = gss(n_splits = 1, test_size = 0.2, random_state = 42)
    train_idx, test_idx = next(splitter.split(df, y, groups = df['session']))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
else:
    train_df, test_df = tts(df, test_size = 0.2, random_state = 42)

#특징 조합별 Random Forest 성능 비교
results = []
for name, columns in feature_sets.items():
    print()
    printer(name)

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f'Missing Columns : {missing}')

    x_train = train_df[columns]
    x_test = test_df[columns]
    y_train = train_df['target_lux']
    y_test = test_df['target_lux']

    #랜덤 포레스트 모델 설정
    model = mp(
        si(strategy = 'median'),
        rfr(n_estimators = 100, random_state = 42, n_jobs = -1)
    )

    #모델 학습
    model.fit(x_train, y_train)

    #예측 수행
    prediction = model.predict(x_test)

    #회귀 성능 평가
    mae = mean_absolute_error(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction) ** 0.5
    r2 = r2_score(y_test, prediction)

    results.append({
        'Feature_Set': name,
        'Feature_Count': len(columns),
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })

    print(f'MAE : {mae:.4f}')
    print(f'RMSE : {rmse:.4f}')
    print(f'R2 : {r2:.4f}')

#특징 조합 비교 결과
printer('Feature Comparison Result')
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by = 'MAE')
print(results_df)

#결과 저장
result_dir = Path('Result')
result_dir.mkdir(parents = True, exist_ok = True)
results_df.to_csv(result_dir / 'random_forest_feature_comparison.csv', index = False)
printer('Results Saved')
print('Best Feature Set : ', results_df.iloc[0]['Feature_Set'])
