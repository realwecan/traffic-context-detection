test_file = '/media/twang/d7034ce1-f34e-49ab-8286-cf9b04027854/dataset/MIO-TCD/VOCMIO/VOCdevkit/VOCMIO/ImageSets/Main/test.txt';
test_file_cache = textread(test_file, '%s','delimiter', '\n');

img_dir = '/media/twang/d7034ce1-f34e-49ab-8286-cf9b04027854/dataset/MIO-TCD/VOCMIO/VOCdevkit/VOCMIO/JPEGImages';

temp_dir = './temp';
result_dir = './results';
final_results_file = './output/context_only_output.csv';

classes = {'articulated_truck', 'bicycle', 'bus', 'car', 'motorcycle', ...
           'motorized_vehicle', 'non-motorized_vehicle', 'pedestrian', ...
           'pickup_truck', 'single_unit_truck', 'work_van'};
      
parfor ii = 1 : length(test_file_cache)
    fprintf('Processing file #%05d...\n', ii);
    fileid = test_file_cache{ii};
    % delete if output file exists
    if exist([result_dir '/' fileid '.txt'], 'file') == 2
        delete([result_dir '/' fileid '.txt']);
    end
    % read in the current neighbour image's size for bbox normalization
    im = imread([img_dir '/' fileid '.jpg']);
    [img_height, img_width, ~] = size(im);    
    % open results file to write
    fid = fopen([result_dir '/' fileid '.txt'],'w');
    for jj = 1 : length(classes) % for each class
        % get training set ground-truths of this class
        cls_grdtr = all_boxes{ii}(all_boxes{ii}(:,5) == jj,:);
        if ~isempty(cls_grdtr) % at least one ground-truths of this class
            append_grdtr = cls_grdtr(:,[1:4 7]);
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