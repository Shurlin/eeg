import numpy as np
from scipy.io import loadmat
from playsound import playsound
from sklearn.pipeline import Pipeline

#models
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from features import Features
from csp_featues import CspFeatures

cf = CspFeatures('wpd') # erp, rfft, wp
X_train_csp, y_train = cf.get_train()

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 数据标准化
    ('svc', SVC())  # 支持向量机分类器
    # ('clf', DecisionTreeClassifier())
    # ('rf', RandomForestClassifier()) # 随机森林
])

pipeline.fit(X_train_csp, y_train)

scores = []

# Predict
for i in cf.tests:
    X_test_csp = cf.get_test(Features(i, True))
    e1_labels = loadmat(f'../data/BCICIV_2a_gdf/true_labels/A0{i}E.mat')
    y_test = e1_labels['classlabel'].reshape(288) - 1

    scores.append(pipeline.score(X_test_csp, y_test))

print(scores)
print(np.mean(scores))

playsound('../sound/lbw.mp3')


# wpd5: [0.4513888888888889, 0.2638888888888889, 0.3993055555555556, 0.2534722222222222, 0.2777777777777778, 0.3194444444444444, 0.3576388888888889, 0.625]
# 0.36848958333333337

# svc
# erp:[0.5173611111111112, 0.2986111111111111, 0.4479166666666667, 0.2604166666666667, 0.2986111111111111, 0.2569444444444444, 0.19791666666666666, 0.4131944444444444]
# 0.3363715277777778
# rfft: [0.3055555555555556, 0.20833333333333334, 0.3333333333333333, 0.2708333333333333, 0.2708333333333333, 0.34375, 0.4791666666666667, 0.3784722222222222]
# 0.3237847222222222
# wp: [0.6041666666666666, 0.3055555555555556, 0.5069444444444444, 0.2743055555555556, 0.3993055555555556, 0.3680555555555556, 0.5138888888888888, 0.6631944444444444]
# 0.45442708333333337
