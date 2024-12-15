function [prediction_svm]=svm(verbose,class_num,classes,svms,kernel_type,trainingVector,trainingVectorLabel,hypervector)

    NumOfClass = class_num;
    num_classes = class_num;
    for k=1:NumOfClass
        if verbose
            fprintf(['Training Classifier ', num2str(classes(k)) ' of ', num2str(num_classes), '\n']);
        end
        class_k_label = trainingVectorLabel == classes(k);
        svms{k} = fitcsvm(trainingVector, class_k_label, 'Standardize',...
            true,'KernelScale', 'auto', 'KernelFunction', kernel_type, ...
            'CacheSize', 'maximal', 'BoxConstraint', 10);
    end
    
    % Classify the test data
    for k=1:NumOfClass
        if verbose
            fprintf(['Classifying with Classifier ', num2str(classes(k)),...
                ' of ', num2str(num_classes), '\n']);
        end
        [~, temp_score] = predict(svms{k}, hypervector);
        score(:, k) = temp_score(:, 2);                     % Her satirin ilgili sutununa sinifla ilgili score degerini diz.
    end
    [~, est_label] = max(score, [], 2);
    h = 610; % 影像的行数
    w = 340; % 影像的列数
    prediction_svm = uint8(zeros(h*w, 1)); % 使用 uint8 代替 im2uint8
    
    for k=1:num_classes
        prediction_svm(find(est_label==k),:) = k;
    end
    
    end