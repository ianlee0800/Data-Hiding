% -------------------------------------------------------------------------
% Copyright (c) 2020
% Rémi Cogranne, UTT (Troyes University of Technology)
% All Rights Reserved.
% -------------------------------------------------------------------------
% This code is provided by the author under GNU GENERAL PUBLIC LICENSE GPLv3
% which, as explained on this webpage
% https://www.gnu.org/licenses/quick-guide-gplv3.html
% Allows modification, redistribution, provided that:
% * You share your code under the same license ;
% * That you give credits to the authors ;
% * The code is used only for Education or Research purposes.
% -------------------------------------------------------------------------

% Only needed for fair execution time comparisons
maxNumCompThreads(1);

%First run seems to be always much slower ... for a meaningful execution time comparison we do a first pre-heating / dry-run
dummyJPEGstruct = jpeg_read( [ '/data/ALASKA_50774_QF75.jpg' ] );
preCover = double(imread( [ '/data/ALASKA_50774.tif' ] ));
[ S_STRUCT , C_STRUCT , pChange , ChangeRate , Deflection ] = SI_MiPOD_fastlog(preCover , dummyJPEGstruct , 0.2);

imgList = dir('/data/*.jpg');
for imgIdx = 1 : numel(imgList) ,
	fprintf('\n  ***** Processing image %s ***** \n' , imgList(imgIdx).name );
	Payload = 0.4;
	%read JPEG struct from JPEG file (only to get a blank struct with Qtables)
	dummyJPEGstruct = jpeg_read( [ imgList(imgIdx).folder '/' imgList(imgIdx).name ] );
	%read preCover images ... whose name is related with the JPEG file ...
	posFS = strfind(imgList(imgIdx).name, '_');
	preCover = double(imread( [ imgList(imgIdx).folder '/' imgList(imgIdx).name(1:posFS(2)-1) '.tif' ] ));

	%get stego DCT coefficients (and Deflection, pChanges and overall ChangeRate)
	tStart = tic;
	[ stegoStruct , coverStruct , pChange , ChangeRate , Deflection ] = SI_MiPOD_fastlog (preCover , dummyJPEGstruct , Payload);
	tEnd = toc(tStart);
	jpeg_write(stegoStruct , [ '/results/' imgList(imgIdx).name ])
	jpeg_write(coverStruct , [ '/results/' imgList(imgIdx).name(1:posFS(2)-1) '_cover.jpg' ])
	StegoDCT = stegoStruct.coef_arrays{1};
	nbnzAC = ( sum( sum( StegoDCT ~= 0) ) - sum( sum( StegoDCT(1:8:end, 1:8:end) ~= 0 ) ) );
	HNats = - pChange(:) .* log( pChange(:) ) -  (1-pChange(:) )  .* log (1-pChange(:) ) ;
	Hbits = -  pChange(:) .* log2( pChange(:) ) -  (1-pChange(:) )  .* log2 (1-pChange(:) ) ;
	fprintf("\t\t\t\t\t\t\t\t  Target payload = %5.2f bits\n", Payload*nbnzAC )
	fprintf("SI-MiPOD fastlog runs in %2.3f sec. \t\t\t\tActual payload  :  %5.2f bits = %5.2f Nats (ternary entropy computed from pChanges) \n" , tEnd , nansum( Hbits ) , nansum( HNats ) )

	tStart = tic;
	[ stegoStruct , coverStruct , pChange , ChangeRate , Deflection  ] = SI_MiPOD_v0(preCover , dummyJPEGstruct , Payload);
	tEnd = toc(tStart);
 	fprintf("SI-MiPOD original (not so fast) runs in %2.3f sec.\t\tActual payload  :  %5.2f bits = %5.2f Nats (ternary entropy computed from pChanges) \n" , tEnd , nansum( -pChange(:) .* log2( pChange(:) ) -(1-pChange(:) )  .* log2 (1-pChange(:) ) ) , nansum( -pChange(:) .* log( pChange(:) ) -(1-pChange(:) )  .* log (1-pChange(:) ) ) )

	tStart = tic;
	[ stegoStruct , coverStruct , pChange ] = SI_EBS(preCover , dummyJPEGstruct , Payload);
	tEnd = toc(tStart);
 	fprintf("SI-EBS runs in %2.3f sec. \t\t\t\t\tActual payload  :  %5.2f bits = %5.2f Nats (ternary entropy computed from pChanges) \n" , tEnd , nansum( -pChange(:) .* log2( pChange(:) ) -(1-pChange(:) )  .* log2 (1-pChange(:) ) ) , nansum( -pChange(:) .* log( pChange(:) ) -(1-pChange(:) )  .* log (1-pChange(:) ) ) )

	tStart = tic;
	[ stegoStruct , coverStruct , pChange ] = SI_UNIWARD(preCover , dummyJPEGstruct , Payload);
	tEnd = toc(tStart);
 	fprintf("SI-UNIWARD runs in %2.3f. \t\t\t\t\tActual payload  :  %5.2f bits = %5.2f Nats (ternary entropy computed from pChanges) \n" , tEnd , nansum( -pChange(:) .* log2( pChange(:) ) -(1-pChange(:) )  .* log2 (1-pChange(:) ) ) , nansum( -pChange(:) .* log( pChange(:) ) -(1-pChange(:) )  .* log (1-pChange(:) ) ) )
end