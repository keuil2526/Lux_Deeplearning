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

#데이터 경로 설정
data_folder = Path('data')
annotation_file = data_folder / 'annotation_master_latest.csv'
feature_file = data_folder / 'feature_master_latest.csv'
merged_file = data_folder / 'lux_data.csv'
data_folder.mkdir(parents = True, exist_ok = True)

#데이터 불러오기
print('Execution Started : ', datetime.now().isoformat(timespec = 'seconds'))
annotation = pd.read_csv(annotation_file)
printer('Annotation Data Loaded')
print('Annotation Shape : ', annotation.shape)

feature = pd.read_csv(feature_file)
printer('Feature Data Loaded')
print('Feature Shape : ', feature.shape)

#현재 분석 파이프라인에서는 특징 데이터를 기준 데이터로 사용
df = feature.copy()
printer('Main Dataset Ready')
print('Data Size : ', df.shape)

#Save_Data
df.to_csv(merged_file, index = False)
printer('Data Saved')
print('Saved : ', merged_file)
