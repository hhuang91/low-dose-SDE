classdef bCTsimu < dynamicprops
    methods(Static)
        function [header,vol] = loaddcmBCT(dirIn,varargin)
            % optional parameters
            defaults.slinums = 'all';
            % read actual settings or set to default if not present
            settings = parseOptions(defaults,varargin);
            slinums = settings.slinums;
            % read the dicom image and the dicom info metadata
            % open the data
            % -- First get the info from the metadata
            if exist(dirIn) ~= 7
                error('Input is not a folder');
            else
                dcmList = dir(fullfile(dirIn,'*.dcm'));
                nFiles = numel(dcmList);
                di = dicominfo([dirIn,'/',dcmList(1).name]);
                header.nf = nFiles;
            end
            % fill common parameters
            header.volSize = [di.Rows, di.Columns, header.nf];
            tmp = sscanf(di.ImageComments,'slice thickness:%f mm; coronal pixel pitch:%fmm');
            header.PixSize = [tmp(2), tmp(2), tmp(1)];
            if nargout<2 % image data is not required, do not read it
                return;
            end
            if isnumeric(slinums) &&...
                    any(slinums>header.nf)
                error(['slinums can''t exceed ' num2str(header.nf)])
            end
            % read in a loop - we assume the volume is ordered
            if strcmp(slinums,'all')
                indx_read = 1:header.nf;
            else
                indx_read = slinums;
            end
            header.volSize(3) = length(indx_read);
            vol = zeros(header.volSize,'single');
            pBar = tqdm('Loading DCMs',length(indx_read));
            for nI = 1:length(indx_read)
                pBar.print(nI)
                tmp = dicomread([dirIn,'/',dcmList(indx_read(nI)).name]);
                vol(:,:,nI) = single(tmp);
            end
            muWater = 0.02;
            muFat = 0.018;
            vol(vol == 1) = muFat;
            vol(vol == 2) = muWater;
            vol(vol == 3) = muWater;
        end
        function hdf5Builder(trainDir,valdnDir,testDir,matInfo,outFN)
            trainList = dir([trainDir,'/*.mat']);
            valdnList = dir([valdnDir,'/*.mat']);
            testList = dir([testDir,'/*.mat']);
            dpF = matInfo.dpF;
            imSize = matInfo.imSize;
            
            trainDatLen = length(trainList)*dpF;
            valdnDatLen = length(valdnList)*dpF;
            testDatLen = length(testList)*dpF;
            
            if ~exist(outFN)
                h5create(outFN,'/TrainPrj',[imSize(:)',trainDatLen],'Datatype','single');
                h5create(outFN,'/ValdnPrj',[imSize(:)',valdnDatLen],'Datatype','single');
                h5create(outFN,'/TestPrj',[imSize(:)',testDatLen],'Datatype','single');
            end
            
            pBar = tqdm('Building Training Data:');
            for indx = 1:length(trainList)
                pBar.print(indx,length(trainList))
                tmpFN = fullfile(trainList(indx).folder,trainList(indx).name);
                saveInd = 1 + (indx-1)*dpF;
                tmpDat = load(tmpFN,'prjs');
                h5write(outFN,'/TrainPrj',permute(tmpDat.prjs,[3,2,1]),[1,1,saveInd],[imSize(:)',dpF]);
            end
            
            pBar = tqdm('Building Validation Data:');
            for indx = 1:length(valdnList)
                pBar.print(indx,length(valdnList))
                tmpFN = fullfile(valdnList(indx).folder,valdnList(indx).name);
                saveInd = 1 + (indx-1)*dpF;
                tmpDat = load(tmpFN,'prjs');
                h5write(outFN,'/ValdnPrj',permute(tmpDat.prjs,[3,2,1]),[1,1,saveInd],[imSize(:)',dpF]);
            end
            
            pBar = tqdm('Building Test Data:');
            for indx = 1:length(testList)
                pBar.print(indx,length(testList))
                tmpFN = fullfile(testList(indx).folder,testList(indx).name);
                saveInd = 1 + (indx-1)*dpF;
                tmpDat = load(tmpFN,'prjs');
                h5write(outFN,'/TestPrj',permute(tmpDat.prjs,[3,2,1]),[1,1,saveInd],[imSize(:)',dpF]);
            end
        end
    end
end