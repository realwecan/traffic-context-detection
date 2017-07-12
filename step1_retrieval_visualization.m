[pdist, pidx] = pdist2(trnfeats, tstfeats, 'cosine', 'Smallest', 1000);

img_dir = '/media/twang/d7034ce1-f34e-49ab-8286-cf9b04027854/dataset/MIO-TCD/VOCMIO/VOCdevkit/VOCMIO/JPEGImages';
train_file = '/media/twang/d7034ce1-f34e-49ab-8286-cf9b04027854/dataset/MIO-TCD/VOCMIO/VOCdevkit/VOCMIO/ImageSets/Main/train.txt';
test_file = '/media/twang/d7034ce1-f34e-49ab-8286-cf9b04027854/dataset/MIO-TCD/VOCMIO/VOCdevkit/VOCMIO/ImageSets/Main/test.txt';

train_file_cache = textread(train_file, '%s','delimiter', '\n');
test_file_cache = textread(test_file, '%s','delimiter', '\n');

figure;

for ii = 1 : 27743
    subplot(5,6,1);
    imshow([img_dir '/' test_file_cache{ii} '.jpg']);
    for jj = 1 : 29
        subplot(5,6,jj+1);
        imshow([img_dir '/' train_file_cache{pidx(jj,ii)} '.jpg']);
    end
    pause;
end