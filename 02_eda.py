from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
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

#데이터 불러오기
print('Execution Started : ', datetime.now().isoformat(timespec = 'seconds'))
df = pd.read_csv('data/lux_data.csv')
printer('Data Load Completed')
print('Data Size : ', df.shape)
print(df.head(5))

#데이터 기본 정보 확인
printer('Data Information')
df.info()
print()
print(df.describe())
print()
print('Missing Values')
print(df.isnull().sum()[lambda x: x > 0])
print('Duplicate Rows : ', df.duplicated().sum())

#Target_Distribution
printer('Target Distribution')
print(df['target_lux'].describe())

plt.figure(figsize = (8,5))
df['target_lux'].hist(bins = 40)
plt.xlabel('Illuminance (lux)')
plt.ylabel('Frequency')
plt.title('Target Lux Distribution')
plt.tight_layout()
plt.savefig(RESULT_DIR / 'target_lux_distribution.png', dpi = 150)
plt.close()

#측정 조건별 조도 분포 확인
if 'label_kind' in df.columns:
    printer('Label Kind Distribution')
    print(df['label_kind'].value_counts(dropna = False))

if 'surface_group_norm' in df.columns:
    printer('Surface Group')
    print(df.groupby('surface_group_norm')['target_lux'].describe())

printer('EDA Completed')
