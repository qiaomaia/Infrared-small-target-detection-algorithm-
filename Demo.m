% This matlab code implements the infrared small target detection model
% based on partial sum of the tensor nuclear norm.
% 
% Reference:
% Zhang, L.; Peng, Z. Infrared Small Target Detection Based on Partial Sum 
% of the Tensor Nuclear Norm. Remote Sens. 2019, 11, 382.
%
% Written by Landan Zhang 
% 2019-2-24
clc;
clear;
close all;
% 开始计时
tic;

addpath('functions/')
addpath('tools/')
saveDir = 'resultWSTNN/';
imgpath = '论文图/';
imgDir = dir([imgpath '4.bmp']);

% patch parameters
patchSize = 40;
slideStep =40;
lambdaL = 0.7;  %tuning

len = length(imgDir);
for i=1:len
    img = imread([imgpath imgDir(i).name]);
    figure,subplot(131)
  imshow(img),title('Original image')

    if ndims( img ) == 3
        img = rgb2gray( img );
    end
    img = double(img);

        

    %% constrcut patch tensor of original image
    tenD = gen_patch_ten(img, patchSize, slideStep);
  
    [n1,n2,n3] = size(tenD);  
   
    %% calculate prior weight map

   r=saliency_map(img);
   priorWeight = mat2gray(r);
   tenW = gen_patch_ten( priorWeight, patchSize, slideStep);
   
% The power p of weighted tensor Schatten p-norm��p in (0,1]
     p=0.7;
for i = 1 :n3
   [U,S,V] = svd(tenD(:,:,i),'econ');
    diagS = diag(S);
end
weightB=10^(1/p).*sqrt(n1^n2)./(diagS+0.00001);

    %% The proposed model
    lambda = lambdaL / sqrt(max(n1,n2)*n3); 
    [tenB, tenT]  = etrpca_tnn_lp(tenD, lambda, tenW,weightB,p);
    %% recover the target and background image
    tarImg = res_patch_ten_mean(tenT, img, patchSize, slideStep);
    backImg = res_patch_ten_mean(tenB, img, patchSize, slideStep);
    bw = bwfunc(tarImg);
    maxv = max(max(double(img)));
    E = uint8( mat2gray(tarImg)*maxv );
    A = uint8( mat2gray(backImg)*maxv );
  
    subplot(132),imshow(E,[]),title('Target image');
    subplot(133),imshow(A,[]),title('Background image');
    imshow(bw);
    % save the results
    %imwrite(E, [saveDir 'target/' imgDir(i).name]);
    %imwrite(A, [saveDir 'background/' imgDir(i).name]);
    %imwrite(bw, [saveDir 'bw/' imgDir(i).name]);
end
toc;