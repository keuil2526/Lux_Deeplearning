from pathlib import Path
import requests
import pandas as pd

#출력 구간 구분
def printer(text):
    print()
    print('='*10, text, '='*10)
    print()

#데이터 다운로드
data_dir = Path('data')
data_dir.mkdir(parents = True, exist_ok = True)
source_file = data_dir / 'feature_master_latest.csv'
annotation_file = data_dir / 'annotation_master_latest.csv'
output_file = data_dir / 'lux_data.csv'
record_url = 'https://zenodo.org/api/records/21514641'

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

printer('Download Dataset')
response = requests.get(record_url, timeout = 30)
response.raise_for_status()
record = response.json()

#필요한 원본 데이터 파일 다운로드
for file_name, save_file in [
    ('feature_master_latest.csv', source_file),
    ('annotation_master_latest.csv', annotation_file)
]:
    if not save_file.exists():
        info = next((item for item in record['files'] if item['key'] == file_name), None)

        if info is None:
            raise FileNotFoundError(f'{file_name} was not found in the Zenodo record.')

        with requests.get(info['links']['self'], stream = True, timeout = 180) as response:
            response.raise_for_status()
            with save_file.open('wb') as file:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        file.write(chunk)

        print('Downloaded : ', save_file)
    else:
        print('Using Existing File : ', save_file)

#분석에 필요한 특징과 타깃 데이터만 선택
df = pd.read_csv(source_file)
required = feature_cols + ['target_lux']

if 'session' in df.columns:
    required.append('session')

missing = [column for column in required if column not in df.columns]
if missing:
    raise KeyError(f'Missing Columns : {missing}')

df = df[required].copy()
df.to_csv(output_file, index = False)

printer('Data Saved')
print('Saved : ', output_file)
print('Data Size : ', df.shape)
