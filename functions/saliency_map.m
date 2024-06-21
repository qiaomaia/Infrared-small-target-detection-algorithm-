
function sm = saliency_map(img, scales)
% 假设已有一个名为'img'的灰度图像，以及一个尺度列表'scales'
Q=[-4,-1,0,-1,-4;
   -1,2,3,2,-1;
    0,3,4,3,0;
   -1,2,3,2,-1;
   -4,-1,0,-1,-4];
img=imfilter(img, Q, 'replicate');
img=img.^2;

% 初始化显著性图，大小与原图相同，值为0
sm = zeros(size(img));

% 对每个尺度进行处理
for d = scales
    [x, y] = size(img);
    
    % 对每个像素位置
    for i = 1:x
        for j = 1:y
            center_pixel = img(i, j);
            
            % 获取并处理子块
            subblock = getSubBlock(img, i, j, d);
            differences = (subblock - center_pixel).^2;
            LE = sum(differences(:));
            LE(LE<=0) = 0; % 确保非负
            
            % 更新显著性图，这里简单累加各尺度的结果
            sm(i, j) = sm(i, j) + LE; 
        end
    end
    
    % 可选：根据尺度大小调整显著性图的贡献（例如，小尺度给予更多权重）
     sm = sm + (d / max(scales)) * LE; 
end

% 最后，可以考虑对sm进行归一化处理，使其值域在一个合适范围内
% sm = sm ./ max(sm(:)); 

% 辅助函数，用于获取指定大小和中心的子块
function sb = getSubBlock(img, centerX, centerY, blockSize)
    halfSize = floor(blockSize/2);
    startX = max(1, centerX-halfSize);
    endX = min(size(img, 1), centerX+halfSize);
    startY = max(1, centerY-halfSize);
    endY = min(size(img, 2), centerY+halfSize);
    sb = img(startX:endX, startY:endY);

