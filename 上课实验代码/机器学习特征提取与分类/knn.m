function [label_test] = knn(k, data_train, label_train, data_test)
    error(nargchk(4,4,nargin));
    %计算出新的特征参数与表3中参数的距离
    dist = l2_distance(data_train, data_test);
    %对距离进行排序
    [sorted_dist, nearest] = sort(dist);
    %选出最近的特征
    nearest = nearest(1:k,:);
    %用最近的特征的故障类型，作为新的特征参数的故障类型
    label_test = label_train(nearest);
end
