from pathlib import Path
import pandas as pd
import sys
import atexit
from datetime import datetime

# 실행 결과를 터미널과 파일에 함께 기록
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return 'utf-8'


RESULT_DIR = Path(__file__).resolve().parent / 'Result'
RESULT_DIR.mkdir(parents = True, exist_ok = True)
result_log = open(RESULT_DIR / f'{Path(__file__).stem}.txt', 'w', encoding = 'utf-8')
sys.stdout = Tee(sys.__stdout__, result_log)
sys.stderr = Tee(sys.__stderr__, result_log)
atexit.register(result_log.close)

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
target_col = 'target_lux'

#데이터 불러오기
print('Execution Started : ', datetime.now().isoformat(timespec = 'seconds'))
df = pd.read_csv('data/lux_data.csv')
printer('Data Load Completed')
print('Data Size : ', df.shape)

#필요한 특징과 타깃 열 확인
required = feature_cols + [target_col]
if 'session' in df.columns:
    required.append('session')

missing = [column for column in required if column not in df.columns]
if missing:
    raise KeyError(f'Missing Columns : {missing}')

df = df[required].copy()

#데이터 형식 변환 및 결측 타깃 제거
for column in feature_cols + [target_col]:
    df[column] = pd.to_numeric(df[column], errors = 'coerce')

before = len(df)
df = df.dropna(subset = [target_col]).reset_index(drop = True)

printer('Preprocessing Completed')
print('Removed Target Rows : ', before - len(df))
print('Missing Feature Values : ', int(df[feature_cols].isnull().sum().sum()))

#전처리 결과 저장
output_file = Path('data/lux_preprocessed.csv')
df.to_csv(output_file, index = False)

printer('Data Saved')
print('Saved : ', output_file)
print('Final Data Size : ', df.shape)
print('Feature Count : ', len(feature_cols))
