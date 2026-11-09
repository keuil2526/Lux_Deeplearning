from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit as gss
from sklearn.model_selection import GroupKFold as gkf
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score as cvs
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
feature_cols = [
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

#전처리 데이터 불러오기
df = pd.read_csv('data/lux_preprocessed.csv')
printer('Data Load Completed')
print(df.head(5))
print('Data Size : ', df.shape)

#입력 특징과 타깃 분리
x = df[feature_cols]
y = df['target_lux']
print(x.columns, x.shape)
print(y.shape)

#학습 데이터와 테스트 데이터 분리
if 'session' in df.columns:
    splitter = gss(n_splits = 1, test_size = 0.2, random_state = 42)
    train_idx, test_idx = next(splitter.split(x, y, groups = df['session']))
    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
else:
    x_train, x_test, y_train, y_test = tts(x, y, test_size = 0.2, random_state = 42)

print('Train Data : ', x_train.shape)
print('Test Data : ', x_test.shape)

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

printer('Model Performance')
print(f'MAE : {mae:.4f}')
print(f'RMSE : {rmse:.4f}')
print(f'R2 : {r2:.4f}')

#교차 검증
if 'session' in df.columns:
    cv = gkf(n_splits = 5)
    scores = cvs(model, x, y, cv = cv, groups = df['session'], scoring = 'r2', n_jobs = -1)
else:
    scores = cvs(model, x, y, cv = 5, scoring = 'r2', n_jobs = -1)

printer('Each Fold R2')
for i, score in enumerate(scores, start = 1):
    print(f'Fold_{i} : {score:.4f}')
print()
print(f'Average_R2 : {scores.mean():.4f}')

#특징 중요도 확인
printer('Feature Importance')
regressor = model.named_steps['randomforestregressor']
importance = pd.DataFrame({
    'Feature': x.columns,
    'Importance': regressor.feature_importances_
})
importance = importance.sort_values(by = 'Importance', ascending = False)
print(importance)

#특징 중요도 시각화
result_dir = Path('Result')
result_dir.mkdir(parents = True, exist_ok = True)

plt.figure(figsize = (8,5))
plt.bar(importance['Feature'].head(15), importance['Importance'].head(15))
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.xticks(rotation = 45)
plt.tight_layout()
plt.savefig(result_dir / 'random_forest_feature_importance.png', dpi = 150)
plt.show()

#결과 저장
pd.DataFrame([{
    'MAE': mae,
    'RMSE': rmse,
    'R2': r2,
    'CV_R2_Mean': scores.mean(),
    'CV_R2_Std': scores.std()
}]).to_csv(result_dir / 'random_forest_results.csv', index = False)
importance.to_csv(result_dir / 'random_forest_feature_importance.csv', index = False)
printer('Results Saved')
