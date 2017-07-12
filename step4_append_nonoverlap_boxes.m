test_file = '/media/twang/d7034ce1-f34e-49ab-8286-cf9b04027854/dataset/MIO-TCD/VOCMIO/VOCdevkit/VOCMIO/ImageSets/Main/test.txt';
test_file_cache = textread(test_file, '%s','delimiter', '\n');
orig_detector_outfile = './output/ssd_512_rescore_approx_output.csv';
% orig_detector_outfile = './output/fasterrcnn_rescore_output.csv';

img_dir = '/media/twang/d7034ce1-f34e-49ab-8286-cf9b04027854/dataset/MIO-TCD/VOCMIO/VOCdevkit/VOCMIO/JPEGImages';

temp_dir = './temp';
result_dir = './results';
final_results_file = './output/ssd_512_appended_approx_output.csv';
% final_results_file = './output/fasterrcnn_appended_output.csv';

classes = {'articulated_truck', 'bicycle', 'bus', 'car', 'motorcycle', ...
           'motorized_vehicle', 'non-motorized_vehicle', 'pedestrian', ...
           'pickup_truck', 'single_unit_truck', 'work_van'};

% a very small value for ill-labeled ground-truth (i.e., x1==x2 or y1==y2)
epsilon = 0.00000001;
       
parfor ii = 1 : length(test_file_cache)
    fprintf('Processing file #%05d...\n', ii);
    fileid = test_file_cache{ii};
    % delete if output file exists
    if exist([result_dir '/' fileid '.txt'], 'file') == 2
        delete([result_dir '/' fileid '.txt']);
    end
    % delete if temp file exists
    if exist([temp_dir '/' fileid '.txt'], 'file') == 2
        delete([temp_dir '/' fileid '.txt']);
    end
    % dump the orginal detection results for this image to a file
    system(['grep "^' fileid '" ' orig_detector_outfile ' >> ' temp_dir '/' fileid '.txt']);
    % check if the detection output for this image is empty
    emptycheck = dir([temp_dir '/' fileid '.txt']);
    if emptycheck.bytes == 0
        continue;
    end
    % read in the current image's size for bbox normalization
    im = imread([img_dir '/' fileid '.jpg']);
    [img_height, img_width, ~] = size(im);    
    % reads in the original detection results
    fid = fopen([temp_dir '/' fileid '.txt']);
    textdata = textscan(fid,'%s %s %f %d %d %d %d','delimiter',',');
    fclose(fid);
    % open results file to write
    fid = fopen([result_dir '/' fileid '.txt'],'w');
    for jj = 1 : length(classes) % for each class
        % get the indices of detections of this class
        box_idx = find(cellfun(@(x) strcmp(x,classes{jj}), textdata{2}));
        % firstly, write detections to results file
        % add their score with -log(epsilon) so that all scores are
        % guaranteed to be positive numbers
        if ~isempty(box_idx)
            for ik = 1 : length(box_idx)
                kk = box_idx(ik);
                fprintf(fid,'%s,%s,%.8f,%d,%d,%d,%d\n', test_file_cache{ii}, ...
                    classes{jj}, double(textdata{3}(kk))-log(epsilon), ...
                    textdata{4}(kk), textdata{5}(kk), textdata{6}(kk), textdata{7}(kk));
            end        
        end
        % get training set ground-truths of this class
        cls_grdtr = all_boxes{ii}(all_boxes{ii}(:,5) == jj,:);
        if ~isempty(cls_grdtr) % at least one ground-truths of this class
            if isempty(box_idx)
                % no detected objects for this class in this image
                nonoverlap_idx = 1 : size(cls_grdtr,1);
            else
                x1 = double(textdata{4}(box_idx)); y1 = double(textdata{5}(box_idx));
                x2 = double(textdata{6}(box_idx)); y2 = double(textdata{7}(box_idx));
                % normalize coordinates by image size
                x1 = x1 ./ img_width; y1 = y1 ./ img_height;
                x2 = x2 ./ img_width; y2 = y2 ./ img_height;
                % prepare two sets of bboxes in the format [x y w h]
                bboxA = [x1 y1 x2-x1 y2-y1];
                bboxB = [cls_grdtr(:,1) cls_grdtr(:,2) ...
                         max(epsilon,cls_grdtr(:,3)-cls_grdtr(:,1)) ...
                         max(epsilon,cls_grdtr(:,4)-cls_grdtr(:,2))];
                % use matlab routine to compute IoU overlap (jaccard index)
                jaccard_idx = bboxOverlapRatio(bboxA, bboxB);
                % find those ground truth boxes whose maximal overlap with
                % any deteced boxes are smaller than 50%
                nonoverlap_idx = find(max(jaccard_idx)<0.5);
            end
            append_grdtr = cls_grdtr(nonoverlap_idx,[1:4 7]);
            append_grdtr(:,1) = ceil(append_grdtr(:,1) .* img_width);
            append_grdtr(:,2) = ceil(append_grdtr(:,2) .* img_height);
            append_grdtr(:,3) = ceil(append_grdtr(:,3) .* img_width);
            append_grdtr(:,4) = ceil(append_grdtr(:,4) .* img_height);            
            % the score is the negative of image similarity
            append_grdtr(:,5) = -append_grdtr(:,5);
            for kk = 1 : size(append_grdtr,1)
                fprintf(fid,'%s,%s,%.8f,%d,%d,%d,%d\n', test_file_cache{ii}, ...
                    classes{jj}, append_grdtr(kk,5), append_grdtr(kk,1), ...
                    append_grdtr(kk,2), append_grdtr(kk,3), append_grdtr(kk,4));
            end
        end
    end
    fclose(fid);
end

% write final results file
if exist(final_results_file, 'file') == 2
    delete(final_results_file);
end
system(['cat ./results/* > ' final_results_file]);