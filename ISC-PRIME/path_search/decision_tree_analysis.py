from cProfile import label
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import svm

from IPython.display import Image
import pydotplus

from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

head_list = ["scene_name", "case_id", "track_id", "s_vx", "s_vy", "s_speed", "s_yaw", "cross_stop_line", "is_veh_front", "min_dist","dist_stop_line", "e_vx", "e_vy", "e_speed", "e_yaw"]
head_dict = {}
for i, key in enumerate(head_list):
    head_dict[key] = i

val_data_df = pd.read_csv("/home/joe/Desktop/Rule-PRIME/RulePRIME/path_search/val_yield_path_more.csv")
train_data_df = pd.read_csv("/home/joe/Desktop/Rule-PRIME/RulePRIME/path_search/train_yield_path_more.csv")

x_train = train_data_df[["s_speed", "is_veh_front", "min_dist", "dist_stop_line"]].values
y_train = train_data_df[["cross_stop_line"]].values

x_val = val_data_df[["s_speed", "is_veh_front", "min_dist", "dist_stop_line"]].values
y_val = val_data_df[["cross_stop_line"]].values


svm_model = svm.SVC(C=5, kernel="poly")
svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_val)
print(classification_report(y_val, y_pred))

# clf = tree.DecisionTreeClassifier(criterion="gini", max_depth=10)
# clf.fit(x_train, y_train)

# dot_data = tree.export_graphviz(clf, out_file=None)

# graph = pydotplus.graph_from_dot_data(dot_data)  
# graph.write_png('/home/joe/Desktop/Rule-PRIME/RulePRIME/path_search/example.png')    #保存图像
# Image(graph.create_png())

# answer = clf.predict(x_val)

# # print(answer)

# # precision, recall, thresholds = precision_recall_curve(y_train, answer)

# # print(precision, recall, thresholds)

# print(classification_report(y_val, answer))



# print("hello")