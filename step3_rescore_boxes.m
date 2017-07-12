test_file = '/media/twang/d7034ce1-f34e-49ab-8286-cf9b04027854/dataset/MIO-TCD/VOCMIO/VOCdevkit/VOCMIO/ImageSets/Main/test.txt';
test_file_cache = textread(test_file, '%s','delimiter', '\n');
orig_detector_outfile = './precomputed/ssd_512_output.csv';
% orig_detector_outfile = './precomputed/fasterrcnn_output.csv';

img_dir = '/media/twang/d7034ce1-f34e-49ab-8286-cf9b04027854/dataset/MIO-TCD/VOCMIO/VOCdevkit/VOCMIO/JPEGImages';

temp_dir = './temp';
result_dir = './results';
final_results_file = './output/ssd_512_rescore_approx_output.csv';
% final_results_file = './output/fasterrcnn_rescore_output.csv';

classes = {'articulated_truck', 'bicycle', 'bus', 'car', 'motorcycle', ...
           'motorized_vehicle', 'non-motorized_vehicle', 'pedestrian', ...
           'pickup_truck', 'single_unit_truck', 'work_van'};

% a very small value for ill-labeled ground-truth (i.e., x1==x2 or y1==y2)
epsilon1 = 0.00000001;
epsilon2 = 0.00000001;
       
% weight parameter for context term (SSD-512, conf_threshold = 0.01)
theta = [0.0075 0.0075 0.005 0.002 0.1 0.0075 0.0075 0.004 0.002 0.0075 0.003];
sqsigma1 = [0.2 0.3 0.4 0.2 0.2 0.6 0.3 0.3 2.5 0.2 0.1].^2;
sqsigma2 = [0.05 100 100 0.05 0.0125 0.05 0.05 0.1 0.025 0.025 100].^2;

% weight parameter for context term (FasterRCNN_threshold=0.001)
% theta = [0.0075 0.0075 0.0075 0.0075 0.1 0.0075 0.1 0.0075 0.003 0.0075 0.0075];
% sqsigma1 = [0.2 0.3 0.3 0.15 0.15 0.15 0.15 0.15 0.3 0.3 0.3].^2;
% sqsigma2 = [0.05 1 0.05 0.1 0.02 0.05 0.02 10 0.02 0.05 0.02].^2;

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
    % read in the current neighbour image's size for bbox normalization
    im = imread([img_dir '/' fileid '.jpg']);
    [img_height, img_width, ~] = size(im);    
    % reads in the original detection results
    fid = fopen([temp_dir '/' fileid '.txt']);
    textdata = textscan(fid,'%s %s %f %d %d %d %d','delimiter',',');
    fclose(fid);
    % open results file to write
    fid = fopen([result_dir '/' fileid '.txt'],'w');
    for jj = 1 : size(textdata{1},1) % for each detection
        [~, cls_idx] = ismember(textdata{2}(jj), classes);
        x1 = double(textdata{4}(jj)); y1 = double(textdata{5}(jj));
        x2 = double(textdata{6}(jj)); y2 = double(textdata{7}(jj));
        % normalize coordinates by image size
        x1 = x1 ./ img_width; y1 = y1 ./ img_height;
        x2 = x2 ./ img_width; y2 = y2 ./ img_height;
        % get training set ground-truths of the same class
        cls_grdtr = all_boxes{ii}(all_boxes{ii}(:,5) == cls_idx,:);
        % prepare two sets of bboxes in the format [x y w h]
        bboxA = [x1 y1 x2-x1 y2-y1];
        bboxB = [cls_grdtr(:,1) cls_grdtr(:,2) ...
                 max(epsilon1,cls_grdtr(:,3)-cls_grdtr(:,1)) ...
                 max(epsilon1,cls_grdtr(:,4)-cls_grdtr(:,2))];
        % use matlab routine to compute IoU overlap (jaccard index)
        jaccard_idx = bboxOverlapRatio(bboxA, bboxB);
        % we define a similarity measure based on jaccard index
        jaccard_sim = exp(-log(jaccard_idx').^2./sqsigma1(cls_idx));
        % we define another similarity measure based on image feature
        % distance (cosine distance, TODO try other distance?)
        imgfeat_sim = exp(-cls_grdtr(:,7).^2/sqsigma2(cls_idx));
        % combine two similarity measures, textdata{3}(jj) is the original
        % detector output score
        combine_score = textdata{3}(jj) + theta(cls_idx) .* log(epsilon2+sum(jaccard_sim .* imgfeat_sim)); 
        fprintf(fid,'%s,%s,%.8f,%d,%d,%d,%d\n', test_file_cache{ii}, ...
            classes{cls_idx}, combine_score, textdata{4}(jj), ...
            textdata{5}(jj), textdata{6}(jj), textdata{7}(jj));
    end
    fclose(fid);
end

% write final results file
if exist(final_results_file, 'file') == 2
    delete(final_results_file);
end
system(['cat ./results/* > ' final_results_file]);