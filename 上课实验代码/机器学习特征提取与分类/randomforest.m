function [label_test] = randomforest(nTrees, features, classLabels, Xts)
B = TreeBagger(nTrees,features,classLabels, 'Method', 'classification');
% Given a new individual WITH the features and WITHOUT the class label,
newData1 = Xts;

% Use the trained Decision Forest.
predChar1 = B.predict(newData1);

% Predictions is a char though. We want it to be a number.
label_test = str2double(predChar1);