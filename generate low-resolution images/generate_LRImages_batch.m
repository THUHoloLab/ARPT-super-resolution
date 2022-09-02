%% generating low resolution images from 
close all;clear all;clc
addpath 'G:\博士后课题\程序\SRToolbox\common';
addpath 'G:\博士后课题\程序\SRToolbox\algorithms\MAP';
% path='G:\博士后课题\程序\生成低分辨图像\LRImages_sets2';
path1='G:\博士后课题\程序\生成低分辨图像\Set12\';

file=dir([path1,'*.png']);
% img0=imread('Set12\12.png');
num_name=length(file);
model = SRModel;
model.magFactor = 0.5;
model.psfWidth = 0.2;
std_noise=0.02;
% savepath=['G:\博士后课题\程序\自适应参数调谐SR\LRImagesSets\Speckle\',num2str(std_noise),'Speckle\'];
% savepath=['G:\博士后课题\程序\自适应参数调谐SR\LRImagesSets\Gaussian\',num2str(std_noise),'Gaussian\'];
savepath=['G:\博士后课题\程序\自适应参数调谐SR\LRImagesSets\3倍\','s',num2str(3),num2str(std_noise),'Gaussian\'];
path_file=savepath(1:end-1);
% mkdir (['G:\博士后课题\程序\自适应参数调谐SR\LRImagesSets\Speckle\',num2str(std_noise),'Speckle']);
% mkdir (['G:\博士后课题\程序\自适应参数调谐SR\LRImagesSets\Gaussian\',num2str(std_noise),'Gaussian'])
mkdir (['G:\博士后课题\程序\自适应参数调谐SR\LRImagesSets\3倍\','s',num2str(3),num2str(std_noise),'Gaussian'])
transform = 'translation'; % 'translation','euclidean','affine','homography'
for name_ind=1:num_name
img0=imread([file(name_ind).folder,'\',file(name_ind).name]);
img0=double(img0(:,:,1));img0=img0./max(img0(:));
% figure,imshow(img0,[]);
iter=100; %generate 30 frames
name_ind
dirname=[file(name_ind).name(1:end-4),'_',num2str(model.psfWidth)];
% mkdir (dirname);%butterfly;%Lines_no_noise1;
H=eye(3);
% H(3)=-1+2*rand;H(6)=-1+2*rand;
W=composeSystemMatrix(size(img0), model.magFactor, model.psfWidth, H');
img_vector=imageToVector(img0);
img_vector_LR=W'*img_vector;
img_LR=vectorToImage(img_vector_LR,round(size(img0).*model.magFactor));
LRImages=zeros([size(img_LR),iter]);
img_LR=img_LR./max(img_LR(:));
LRImages(:,:,1)=imnoise(img_LR,'gaussian',0,std_noise^2);
% LRImages(:,:,1)=imnoise(img_LR,'speckle',std_noise^2);
for i=2:iter
H=eye(3);
% H(3)=-2+2*rand;H(6)=-2+2*rand;
H(3)=-1+2*rand;H(6)=-1+2*rand;

W=composeSystemMatrix(size(img0), model.magFactor, model.psfWidth, H');
img_vector=imageToVector(img0);
img_vector_LR=W'*img_vector;
img_LR=vectorToImage(img_vector_LR,round(size(img0).*model.magFactor));
% img_LR=imtranslate('');
img_LR=img_LR./max(img_LR(:));
img_LR=imnoise(img_LR,'gaussian',0,std_noise^2);
% img_LR=imnoise(img_LR,'speckle',std_noise^2);
% figure,imshow(imnoise(img_LR,'gaussian',0.01));
% img_LR=imtranslate(img_LR,[randi([-1,1],1,1).*H(3),randi([-1,1],1,1).*H(6)]);

% img_LR=imtranslate(img0,[H(3),H(6)]);
% img_LR=imresize(img_LR,0.5);

% for i=1:iter
% img_LR0=imtranslate(img_LR,randi([-1,1],1,2).*0.5*rand);
% figure,imshow(img_LR,[]);
% imwrite(imresize(img_LR,2),['Lines_no_noise2\LRImages',num2str(i),'.jpg']);
% imwrite(img_LR,['Barbara1\LRImages',num2str(i),'.jpg']);
% imwrite(img_LR,['butterfly1',num2str(model.psfWidth),'\LRImages',num2str(i),'.jpg']);
LRImages(:,:,i)=img_LR;
end
% save([savepath,dirname,'psf_LRImages_',num2str(std_noise),'noise_',num2str(iter),'frame','.mat'],'LRImages')
% save([savepath,dirname,'psf_LRImages_',num2str(std_noise),'gaussian_',num2str(iter),'frame','.mat'],'LRImages')
save([savepath,dirname,'s3_psf_LRImages_',num2str(std_noise),'gaussian_',num2str(iter),'frame','.mat'],'LRImages')
% save([savepath,dirname,'psf_LRImages_',num2str(std_noise),'speckle_',num2str(iter),'frame','.mat'],'LRImages')

end