function [ S_STRUCT , C_STRUCT , pChange , ChangeRate , Deflection ] = SI_MiPOD_fastlog (preCover, C_STRUCT, Payload)
% -------------------------------------------------------------------------
% SI_MiPOD Embedding       |      May 2020       |      version 1.0 
% -------------------------------------------------------------------------
% INPUT:
%  - Cover       - Path to the cover image or the cover image or JPEG STRUCT
%  - Payload     - Embedding payload in bits per DCT coefs (bpc).
% OUTPUT:
%  - S_STRUCT    - Resulting stego jpeg STRUCT with embedded payload
%  - FisherInfo  - Fisher Info for each and every coefs. 
%  - Deflection  - Overall deflection. 
%  - pChange     - Embedding change probabilities. 
%  - ChangeRate  - Average number of changed pixels
% -------------------------------------------------------------------------
% Copyright (c) 2020 
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from DDE Lab. DDE Lab does not warrant the
% operation of the program will be uninterrupted or error-free. The
% end-user understands that the program was developed for research purposes
% and is advised not to rely exclusively on the program for any reason. In
% no event shall Binghamton University or DDE Lab be liable to any party
% for direct, indirect, special, incidental, or consequential damages,
% including lost profits, arising out of the use of this software. DDE Lab
% disclaims any warranties, and has no obligations to provide maintenance,
% support, updates, enhancements or modifications.
% -------------------------------------------------------------------------
% Contact: remi.cogranne@utt.fr
%          January 2020
% -------------------------------------------------------------------------
% -------------------------------------------------------------------------
% Read the JPEG image if needed
if ischar( preCover )
    preCover = double(imread( preCover ));
end

% Read the JPEG image if needed
if ischar( C_STRUCT )
    C_STRUCT = jpeg_read( C_STRUCT );
end


% tStart0 = tic;
% tStart = tic;
C_QUANT = C_STRUCT.quant_tables{1};

%First let us get the DCT matrix
MatDCT = RCgetJPEGmtx;
DCT_real = vec2im( MatDCT *  im2vec( preCover - 128 , [8,8] )  ./C_QUANT(:) , [0,0], [8,8]) ;
DCT_rounded = round(DCT_real);


% get SI, that is the real-valued DCT values from the uncompressed image
C_STRUCT.coef_arrays{1} = DCT_rounded;
e = DCT_rounded - DCT_real;             % Compute rounding error
sgn_e = sign(e);
sgn_e(e==0) = round(rand(sum(e(:)==0), 1)) * 2 - 1;
change = - sgn_e;


% Compute Variance in spatial domain ....
% tStart = tic;
WienerResidual = preCover - wiener2(preCover,[2,2]);
Variance = VarianceEstimationDCT2D(WienerResidual,3,3);
% fprintf('Variance estimation (spatial domain) takes %5.5f \n ' , toc(tStart) ),

% tStart = tic;
% ... and apply the covariance transformation to DCT domain
% funVar = @(x) reshape( diag(MatDCT*diag(x(:))*MatDCT')  , 8 , 8 ) ./ ( C_STRUCT.quant_tables{1}.^2 );
% VarianceDCT = blkproc(Variance,[8 8],funVar);

% In this code we replaced the blkproc with nested loops and simplied covariance linear transformation
MatDCTq = MatDCT.^2;
Qvec = C_STRUCT.quant_tables{1}(:).^2;
for idx=1:64 , MatDCTq(idx,:) = MatDCTq(idx,:)./ Qvec(idx); end

VarianceDCT = zeros(size(preCover));
for idxR=1:8:size( Variance, 1)
    for idxC=1:8:size( Variance, 2)
        tmp = Variance(idxR:idxR+7 , idxC:idxC+7);
        VarianceDCT(idxR:idxR+7 , idxC:idxC+7) = reshape( MatDCTq * tmp(:) , 8,8);
    end
end
VarianceDCT(VarianceDCT<1e-10) = 1e-10;
% fprintf('Variance estimation (JPEG domain) takes %5.5f \n ' , toc(tStart) ),
% Compute Fisher information and smooth it
% tStart = tic;
FisherInformation = 1 ./ VarianceDCT.^2;

%Post Filter
tmp = zeros(size( FisherInformation ) + 16);
tmp(9:end-8, 9:end-8) = FisherInformation;
tmp(1:8, :) = tmp(9:16, :);
tmp(end-7:end, :) = tmp(end-15:end-8, :);
tmp( : , 1:8, :) = tmp( : , 9:16);
tmp( : , end-7:end, :) = tmp( : , end-15:end-8);
FisherInformation = ( tmp(1:end-16 , 1:end-16) + tmp(9:end-8 , 1:end-16) * 3 + tmp(17:end , 1:end-16) + tmp(1:end-16 , 9:end-8) * 3 + tmp(9:end-8 , 9:end-8) * 4 + tmp(17:end , 9:end-8) * 3 + tmp(1:end-16 , 17:end) + tmp(9:end-8 , 17:end) * 3 + tmp(17:end , 17:end) ) / 20 ; 
%     end


% Compute embedding change probabilities and execute embedding
FI = FisherInformation .* (2*e-sgn_e).^4;
maxCostMat = false(size(FI));
maxCostMat(1:8:end, 1:8:end) = true;
maxCostMat(5:8:end, 1:8:end) = true;
maxCostMat(1:8:end, 5:8:end) = true;
maxCostMat(5:8:end, 5:8:end) = true;
FI(maxCostMat & (abs(e)>0.4999)) = 1e10;
FI(abs(e)<0.01) = 1e10;
FI = FI(:)';
% fprintf('Variance smoothing takes %5.5f \n ' , toc(tStart) ),

S_COEFFS = C_STRUCT.coef_arrays{1};

% Ternary embedding change probabilities
nzAC = sum(S_COEFFS(:)~=0) - sum(sum(sum(S_COEFFS(1:8:end,1:8:end,:)~=0 ) ) );
messageLenght = round(Payload * nzAC * log(2));

% tStart = tic;
[ beta ] = BinaryProbs(FI,messageLenght);
% fprintf('Embedding probabilities takes = %5.5f \n ' , toc(tStart) ),

% Simulate embedding
%beta = 2 * beta;
% tStart = tic;
RandStream.setGlobalStream(RandStream('mlfg6331_64','NormalTransform','Polar'))
r = rand(1,numel(S_COEFFS));
ModifPM1 = (r < beta);                % Cover elements to be modified by +-1
S_COEFFS(ModifPM1) = S_COEFFS(ModifPM1) + change(ModifPM1); % Modifying X by +-1
S_COEFFS(S_COEFFS>1024) = 1024;                    % Taking care of boundary cases
S_COEFFS(S_COEFFS<-1023)   = -1023;
ChangeRate = sum(ModifPM1(:))/numel(S_COEFFS); % Computing the change rate
pChange = reshape(beta,size(S_COEFFS));

%
S_STRUCT = C_STRUCT;
S_STRUCT.coef_arrays{1} = S_COEFFS;
Deflection = sum( pChange(:) .* FI(:) );
% fprintf('Embedding simuation takes = %5.5f \n ' , toc(tStart) ),

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Beginning of the supporting functions %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ dct8_mtx ] = RCgetJPEGmtx
    [cc,rr] = meshgrid(0:7);
    T = sqrt(2 / 8) * cos(pi * (2*cc + 1) .* rr / (2 * 8));
    T(1,:) = T(1,:) / sqrt(2);
    dct8_mtx = zeros(64,64);
    for i=1:64 ; dcttmp=zeros(8); dcttmp(i)=1; TTMP =  T*dcttmp*T'; dct8_mtx(:,i) = TTMP(:); end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation of the pixels' variance based on a 2D-DCT (trigonometric polynomial) model
function EstimatedVariance = VarianceEstimationDCT2D(Image, BlockSize, Degree)
% verifying the integrity of input arguments
    if ~mod(BlockSize,2)
        error('The block dimensions should be odd!!');
    end
    if (Degree > BlockSize)
        error('Number of basis vectors exceeds block dimension!!');
    end

    % number of parameters per block
    q = Degree*(Degree+1)/2;

    % Build G matirx
    BaseMat = zeros(BlockSize);BaseMat(1,1) = 1;
    G = zeros(BlockSize^2,q);
    k = 1;
    for xShift = 1 : Degree
        for yShift = 1 : (Degree - xShift + 1)
            G(:,k) = reshape(idct2(circshift(BaseMat,[xShift-1 yShift-1])),BlockSize^2,1);
            k=k+1;
        end
    end

    % Estimate the variance
    PadSize = floor(BlockSize/2*[1 1]);
    I2C = im2col(padarray(Image,PadSize,'symmetric'),BlockSize*[1 1]);
    PGorth = eye(BlockSize^2) - (G*((G'*G)\G'));
    EstimatedVariance = reshape(sum(( PGorth * I2C ).^2)/(BlockSize^2 - q),size(Image));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computing the embedding change probabilities
function [beta , M] = BinaryProbs(FI, payload)
    load('ixlnx2_logscale.mat');
    % Absolute payload in nats
    % Initial search interval for lambda
    [L, R] = deal (10000 , 60000);

    fL = hBinary(invxlnx2_fast(L*FI,xlog,ylog)) - payload;
    fR = hBinary(invxlnx2_fast(R*FI,xlog,ylog)) - payload;
    % If the range [L,R] does not cover alpha enlarge the search interval
    while fL*fR > 0
        if fL > 0
            L = R;
            fL = fR;
            R = 2*R;
            fR = hBinary(invxlnx2_fast(R*FI,xlog,ylog)) - payload;
    % fprintf("increase \n");
        else
            R = L;
            fR = fL;
            L = L/2;
            fL = hBinary(invxlnx2_fast(L*FI,xlog,ylog)) - payload;
    % fprintf("decrease \n");
        end
    end

    % Search for the labmda in the specified interval
    i=0;
    M = (L+R)/2;
    fM = hBinary(invxlnx2_fast(M*FI,xlog,ylog)) - payload;
    while (abs(fM)>max(2,payload/1000.0) && i<20)
        if fL*fM < 0
            R = M; fR = fM;
        else
            L = M; fL = fM;
        end
        i = i + 1;
        M = (L+R)/2;
        fM = hBinary(invxlnx2_fast(M*FI,xlog,ylog)) - payload;
    end
    % Compute beta using the found lambda
    beta = invxlnx2_fast(M*FI,xlog,ylog);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function x = invxlnx2_fast(y,xlog,ylog)
% Fast solving y = 1/x*log(1/x-1) for x, y can be a vector but not a matrix
% x is the change rate, y is I*lambda, where I is Fisher info
% tStart = tic;
    x=zeros(size(y));
    i_large = y>1000;
    if sum(i_large(:)) > 0
        yz=y(i_large);
        z = yz./log(yz-1);
        for j = 1 : 3
            z = yz./log(z-1);
        end
        x(i_large) = 1./z;
    end

% fprintf('One 1/2 iteration of function "invxlnx3_fast" takes \t %5.5f \n ' , toc(tStart) ),
    i_small = y<=1000;
    if sum(i_small(:)) > 0
        z = y(i_small);
        indlog=floor(( log2(z) +25)/0.02)+1;
        indlog(indlog<1)=1;
        x(i_small) = ylog(indlog) + (z-xlog(indlog))./(xlog(indlog+1)-xlog(indlog)).*(ylog(indlog+1)-ylog(indlog));
    end
    x(isnan(x))=0;
% fprintf('One new iteration of function "invxlnx3_fast" takes \t %5.5f \n ' , toc(tStart) ),
% fprintf('small_size =  %5.5f -- large_size = %5.5f \n ' , sum(i_small(:)) , sum(i_large(:)) ),
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ternary entropy function expressed in nats
function Ht = hBinary(Probs)
% tStart = tic;
    H = -Probs.*log(Probs) - (1-Probs).*log(1-Probs) ;
    H(Probs<1e-10)=0;
    H(Probs>1-1e-10)=0;
    H(isnan(Probs))=0;
    Ht = sum(H);
% fprintf('One new iteration of function "hBinary" takes \t %5.5f \n ' , toc(tStart) ),
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Source code used to pre-compute "ixlnx2_logscale.mat" %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
