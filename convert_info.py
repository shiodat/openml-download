# coding: utf-8

df = pd.read_csv('data-info.csv')

def feature_type(row):
    if row['NumberOfNumericFeatures'] == row['NumberOfFeatures']:
        return 'number'
    elif row['NumberOfSymbolicFeatures'] == row['NumberOfFeatures']:
        return 'category'
    else:
        return 'mix'
    
def convert_numeric_features(row):
    if row['NumberOfClasses'] == -1:
        return row['NumberOfNumericFeatures'] - 1
    else:
        return row['NumberOfNumericFeatures']

def convert_symbolic_features(row):
    if row['NumberOfClasses'] == -1:
        return row['NumberOfSymbolicFeatures']
    else:
        return row['NumberOfSymbolicFeatures'] - 1
    
def ml_type(row):
    if row['NumberOfClasses'] == -1:
        return 'regression'
    else:
        return 'classification'
        
df['NumberOfFeatures'] = df['NumberOfFeatures'] - 1
df['NumberOfSymbolicFeatures'] = df.apply(lambda row: convert_symbolic_features(row), axis=1)
df['NumberOfNumericFeatures'] = df.apply(lambda row: convert_numeric_features(row), axis=1)
df['特徴量型'] = df.apply(lambda row: feature_type(row), axis=1)
df['機械学習タイプ'] = df.apply(lambda row: ml_type(row), axis=1)
df['欠損値の有無'] = df['NumberOfMissingValues'].apply(lambda x: x != 0)
df['ラベルの偏り'] = (df['MajorityClassSize'] - df['MinorityClassSize']) / df['NumberOfInstances']
df['欠損レコード率'] = df['NumberOfInstancesWithMissingValues'] / df['NumberOfInstances']

df = df.drop(['format', 'status', 'MaxNominalAttDistinctValues'], axis=1)

df = df.rename(columns={
    'did': 'データID',
    'name': 'データ名',
    'MajorityClassSize': '最大ラベル数',
    'MinorityClassSize': '最小ラベル数',
    'NumberOfClasses': 'クラス数',
    'NumberOfFeatures': '特徴量数',
    'NumberOfInstances': 'レコード数',
    'NumberOfInstancesWithMissingValues': '欠損ありレコード数',
    'NumberOfMissingValues': '欠損セル数',
    'NumberOfNumericFeatures': '数値特徴量数',
    'NumberOfSymbolicFeatures': 'カテゴリ特徴量数'
})

df.head()
df.to_csv('data_info2.csv', index=False, encoding='utf-8')