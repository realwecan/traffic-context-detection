% for each test set image, collect bounding boxes from their
% neighbours in the training set

% number of neighbours for each image
num_nb = 50;

gt_train_file = '/home/twang/Desktop/MIO-TCD-Localization-Code/gt_train.csv';

img_dir = '/media/twang/d7034ce1-f34e-49ab-8286-cf9b04027854/dataset/MIO-TCD/VOCMIO/VOCdevkit/VOCMIO/JPEGImages';

train_file = '/media/twang/d7034ce1-f34e-49ab-8286-cf9b04027854/dataset/MIO-TCD/VOCMIO/VOCdevkit/VOCMIO/ImageSets/Main/train.txt';
train_file_cache = textread(train_file, '%s','delimiter', '\n');

classes = {'articulated_truck', 'bicycle', 'bus', 'car', 'motorcycle', ...
           'motorized_vehicle', 'non-motorized_vehicle', 'pedestrian', ...
           'pickup_truck', 'single_unit_truck', 'work_van'};

boxes_dir = './boxes';
temp_dir = './boxes_temp';

% cache training bounding boxes to disk
parfor ii = 1 : 99000
    fileid = train_file_cache{ii};
    filename = [temp_dir '/' fileid '.txt'];
    if exist(filename, 'file')==2
        delete(filename);
    end
    system(['grep "^' fileid '" ' gt_train_file ' >> ' temp_dir '/' fileid '.txt']);    
end

% map
parfor ii = 1 : 27743
    fprintf('Processing file #%05d...\n', ii);
    if exist([boxes_dir '/' int2str(ii) '.txt'], 'file') == 2
        delete([boxes_dir '/' int2str(ii) '.txt']);
    end
    ffid = fopen([boxes_dir '/' int2str(ii) '.txt'], 'w');
    for jj = 1 : num_nb
        fileid = train_file_cache{pidx(jj,ii)};
        % check if the this neighbour has no bounding boxes at all
        emptycheck = dir([temp_dir '/' fileid '.txt']);
        if emptycheck.bytes == 0
            continue;
        end
        % read in the current neighbour image's size for bbox normalization
        im = imread([img_dir '/' fileid '.jpg']);
        [img_height, img_width, ~] = size(im);
        % normalize & cache all bboxes to all_boxes
        fid = fopen([temp_dir '/' fileid '.txt']);
        textdata = textscan(fid,'%s %s %d %d %d %d','delimiter',',');
        fclose(fid);
        for kk = 1 : size(textdata{1},1) % number of detections
            [~, cls_idx] = ismember(textdata{2}(kk), classes);
            x1 = double(textdata{3}(kk)); y1 = double(textdata{4}(kk));
            x2 = double(textdata{5}(kk)); y2 = double(textdata{6}(kk));
            % normalize coordinates by image size
            x1 = x1 ./ img_width; y1 = y1 ./ img_height;
            x2 = x2 ./ img_width; y2 = y2 ./ img_height;
            % save bbox location plus the image's index and pairwise
            % distance between the train/test image
            fprintf(ffid,'%f,%f,%f,%f,%d,%d,%f\n',x1,y1,x2,y2,cls_idx,pidx(jj,ii),pdist(jj,ii));
        end
    end
    fclose(ffid);
end

% reduce
all_boxes = cell(0);
for ii = 1 : 27743
    all_boxes{ii} = dlmread([boxes_dir '/' int2str(ii) '.txt']);
end
