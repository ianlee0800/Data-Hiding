function [ S_STRUCT , C_STRUCT , pChange , ChangeRate , Deflection ] = SI_MiPODv0 (preCover, C_STRUCT, Payload)
% -------------------------------------------------------------------------
% SI_MiPOD Embedding       |      January 2020       |      version 0.1 
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

if (nargin < 4)
	Wie=2;
end
if (nargin < 5)
	BlkSz=3;
end
if (nargin < 6)
	Degree=3;
end
if (nargin < 7)
	F = [  1 3 1 ; 3 2 3 ; 1 3 1  ] ; F = F/sum(F(:));
end

wetConst = 1e10;

%cover_STRUCT = jpeg_read(JPEGcoverPath);
C_QUANT = C_STRUCT.quant_tables{1};
%C_STRUCT = cover_STRUCT;
%preCover = double(imread(precoverPath));

fun=@dct2;
xi= blkproc(double(preCover)-128,[8 8],fun);
% Quantization
fun = @(x) x./C_QUANT;
DCT_real = blkproc(xi,[8 8],fun);
DCT_rounded = round(DCT_real);

C_STRUCT.coef_arrays{1} = DCT_rounded;

e = DCT_rounded - DCT_real;             % Compute rounding error
sgn_e = sign(e);
sgn_e(e==0) = round(rand(sum(e(:)==0), 1)) * 2 - 1;
change = - sgn_e;






%First let us get the DCT matrix
MatDCT = RCgetJPEGmtx;

nb_coul = numel( C_STRUCT.coef_arrays );
% Compute Variance and do the flooring for numerical stability
VarianceDCT = zeros(size(preCover));
for cc = 1:nb_coul
    WienerResidualCC = preCover(:,:,cc) - wiener2(preCover(:,:,cc),[Wie,Wie]);
    VarianceCC = VarianceEstimationDCT2D(WienerResidualCC,BlkSz,Degree);
    if (cc==1)
        funVar = @(x) reshape( diag(MatDCT*diag(x(:))*MatDCT')  , 8 , 8 ) ./ ( C_STRUCT.quant_tables{1}.^2 );
    else
        funVar = @(x) reshape( diag(MatDCT*diag(x(:))*MatDCT')  , 8 , 8 ) ./ ( C_STRUCT.quant_tables{2}.^2 );        
    end
    VarianceDCT(:,:,cc) = blkproc(VarianceCC,[8 8],funVar);
end
VarianceDCT(VarianceDCT<sqrt(1/wetConst)) = sqrt(1/wetConst);



% Compute Fisher information and (do not) smooth it
% FisherInformation = imfilter(FisherInformation,fspecial('average',7),'symmetric');
FisherInformation = 1 ./VarianceDCT.^2;

%Post Filter
for cc=1:nb_coul
    tmp = zeros(size( FisherInformation(:,:,cc) ) + 16);
    tmp(9:end-8, 9:end-8) = FisherInformation(:,:,cc);
    tmp(1:8, :) = tmp(9:16, :);
    tmp(end-7:end, :) = tmp(end-15:end-8, :);
    tmp( : , 1:8, :) = tmp( : , 9:16);
    tmp( : , end-7:end, :) = tmp( : , end-15:end-8);
    FisherInformation(:,:,cc) =  tmp(1:end-16 , 1:end-16) * F(1,1) + tmp(9:end-8 , 1:end-16) * F(2,1)  + tmp(17:end , 1:end-16) * F(3,1) + tmp(1:end-16 , 9:end-8) * F(1,2) + tmp(9:end-8 , 9:end-8) * F(2,2) + tmp(17:end , 9:end-8) * F(3,2) + tmp(1:end-16 , 17:end) * F(1,3) + tmp(9:end-8 , 17:end) * F(2,3) + tmp(17:end , 17:end) * F(3,3) ; 
end


% Compute embedding change probabilities and execute embedding
FI = FisherInformation .* (2*e-sgn_e).^2;
maxCostMat = false(size(FI));
maxCostMat(1:8:end, 1:8:end) = true;
maxCostMat(5:8:end, 1:8:end) = true;
maxCostMat(1:8:end, 5:8:end) = true;
maxCostMat(5:8:end, 5:8:end) = true;
FI(maxCostMat & (abs(e)>0.4999)) = wetConst;
FI(abs(e)<0.01) = wetConst;
FI = FI(:)';


S_COEFFS = zeros(size( VarianceDCT ) );
for cc=1:nb_coul
    S_COEFFS(:,:,cc) = C_STRUCT.coef_arrays{cc};
end



% Ternary embedding change probabilities
nzAC = sum(S_COEFFS(:)~=0) - sum(sum(sum(S_COEFFS(1:8:end,1:8:end,:)~=0 ) ) );
messageLenght = round(Payload * nzAC * log(2));


[ beta] = TernaryProbs(FI,messageLenght);

% Simulate embedding
%beta = 2 * beta;
r = rand(1,numel(S_COEFFS));
ModifPM1 = (r < beta);                % Cover elements to be modified by +-1
S_COEFFS(ModifPM1) = S_COEFFS(ModifPM1) + change(ModifPM1); % Modifying X by +-1
S_COEFFS(S_COEFFS>1024) = 1024;                    % Taking care of boundary cases
S_COEFFS(S_COEFFS<-1023)   = -1023;
ChangeRate = sum(ModifPM1(:))/numel(S_COEFFS); % Computing the change rate
pChange = reshape(beta,size(S_COEFFS));

%

S_STRUCT = C_STRUCT;
for cc=1:nb_coul
    S_STRUCT.coef_arrays{cc} = S_COEFFS(:,:,cc);
end

Deflection = sum( pChange(:) .* FI(:) );

end

% Beginning of the supporting functions


function [ dct8_mtx ] = RCgetJPEGmtx
    [cc,rr] = meshgrid(0:7);
    T = sqrt(2 / 8) * cos(pi * (2*cc + 1) .* rr / (2 * 8));
    T(1,:) = T(1,:) / sqrt(2);
    dct8_mtx = zeros(64,64);
    for i=1:64 ; dcttmp=zeros(8); dcttmp(i)=1; TTMP =  T*dcttmp*T'; dct8_mtx(:,i) = TTMP(:); end
end

function [ imDecompress , dct8_mtx ] = RCdecompressJPEG(imJPEG)

nb_coul = numel( imJPEG.coef_arrays );

[cc,rr] = meshgrid(0:7);
T = sqrt(2 / 8) * cos(pi * (2*cc + 1) .* rr / (2 * 8));
T(1,:) = T(1,:) / sqrt(2);


dct8_mtx = zeros(64,64);
for i=1:64 ; dcttmp=zeros(8); dcttmp(i)=1; TTMP =  T*dcttmp*T'; dct8_mtx(:,i) = TTMP(:); end

imDecompress = zeros( [ size( imJPEG.coef_arrays{1} ), numel( imJPEG.coef_arrays ) ] );

for cc=1:nb_coul
    DCTcoefs = imJPEG.coef_arrays{cc};
    if cc==1
        QM = imJPEG.quant_tables{cc};
    else
        QM = imJPEG.quant_tables{2};
    end
    funIDCT = @(x) T'*(x.*QM)*T ;
    imDecompress(:,:,cc) = blkproc(DCTcoefs,[8 8],funIDCT);
end

end

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

% Computing the embedding change probabilities
function [beta] = TernaryProbs(FI, payload)

load('ixlnx2.mat');

% Absolute payload in nats

% Initial search interval for lambda
[L, R] = deal (5*10^-1.5 , 5*10^0.5);

fL = hBinary(invxlnx2_fast(L*FI,xx,yy)) - payload;
fR = hBinary(invxlnx2_fast(R*FI,xx,yy)) - payload;
% If the range [L,R] does not cover alpha enlarge the search interval
while fL*fR > 0
    if fL > 0
        R = 2*R;
        fR = hBinary(invxlnx2_fast(R*FI,xx,yy)) - payload;
% fprintf("increase \n");
    else
        L = L/2;
        fL = hBinary(invxlnx2_fast(L*FI,xx,yy)) - payload;
% fprintf("decrease \n");
    end
end

% Search for the labmda in the specified interval
[i, fM, TM] = deal(0, 1, zeros(20,2));
while (abs(fM)>0.001 && i<20)
    M = (L+R)/2;
%     fprintf("tmp --> M=%5.5f \n",M);
    fM = hBinary(invxlnx2_fast(M*FI,xx,yy)) - payload;
    if fL*fM < 0, R = M; fR = fM;
    else          L = M; fL = fM; end
    i = i + 1;
    TM(i,:) = [fM,M];
end
if (i==20)
    M = TM(find(abs(TM(:,1)) == min(abs(TM(:,1))),1,'first'),2);
end
% Compute beta using the found lambda
beta = invxlnx2_fast(M*FI,xx,yy);
% fprintf("M(final)=%5.5f \n",M);
end



function x = invxlnx2_fast(y,xx,yy)
% Fast solving y = 1/x*log(1/x-1) for x, y can be a vector but not a matrix
% x is the change rate, y is I*lambda, where I is Fisher info
x=zeros(size(y));
i_large = y>1000;
if sum(i_large(:)) > 0
    z = y(i_large)./log(y(i_large)-1);

    for j = 1 : 20
        z = y(i_large)./log(z-1);
    end
    x(i_large) = 1./z;
end

i_small = y<=1000;
if sum(i_small(:)) > 0
    z = y(i_small);
    N = numel(xx);
    M = numel(z);
    comparison = z(:)*ones(1,N) >= ones(M,1)*xx;
    ind = sum(comparison,2)';
    x(i_small) = yy(ind) + (z-xx(ind))./(xx(ind+1)-xx(ind)).*(yy(ind+1)-yy(ind));
end
end

% Ternary entropy function expressed in nats
function Ht = hBinary(Probs)

p0 = 1-Probs;
P = [p0(:);Probs(:)];
H = -(P .* log(P));
H((P<eps)) = 0;
Ht = nansum(H);

end
