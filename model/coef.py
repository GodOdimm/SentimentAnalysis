PATH = {
    # 'EMO_ANNOTATIONS': 'D:\数据集\emoMusic\\annotations',
    'DEAM_ANNOTATIONS': 'D:\DataSet\DEAM\\annotations\\annotations averaged per song\\song_level',
    'DEAM_FEATURE_PATH': 'D:\\DataSet\\DEAM\\DeamCsvPythonFeatures',
    # 'EMO_FEATURES_PATH': 'D:\数据集\emoMusic\\45fatures2',
    'DEAM_ORIGIN_FEATURES_PATH': 'D:\\DataSet\\DEAM\\features'
}

TRAIN_COEF = {
    'epoch': 40,
    'batch_size': 64,
    'drop_out': 0.1,
    'test_size': 0.2,
    'shape': (3800, 111)
}
