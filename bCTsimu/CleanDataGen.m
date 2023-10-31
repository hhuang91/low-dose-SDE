classdef CleanDataGen < bCTsimu
    properties
        dirList
        recDir
        GPUid
        jID
        iD
        datType
        nInst
        header
        vol
        outDir
        pm
        detInfo
    end
    methods
        function obj = CleanDataGen(datType,nInst,GPUid,device,iD,jID)
            trainNum = 120;
            valdnNum = 15;
            testNum = 15;
            obj.GPUid = GPUid;
            obj.datType = datType;
            obj.nInst = nInst;
            obj.iD = iD;
            obj.jID = jID;
            if strcmpi('cluster',device)
                recBaseDir = '/mnt/blackhole-data_drobo/20220715_bCTnoiseSim/bCT_Phantoms';
            else
                recBaseDir = '\\istar-blackhole\data_drobo\20220715_bCTnoiseSim\bCT_Phantoms';
            end
            obj.recDir = recBaseDir;
            obj.dirList = dir(obj.recDir);
            dirFlags = [obj.dirList.isdir];
            obj.dirList = obj.dirList(dirFlags);
            if strcmpi(datType,'Train')
                obj.recDir = [obj.recDir,'/',obj.dirList( mod(obj.iD,trainNum) + 3 ).name];
                %obj.dirList = dir(obj.recDir);
            elseif strcmpi(datType,'Valdn')
                obj.recDir = [obj.recDir,'/',obj.dirList( mod(obj.iD,valdnNum) + trainNum + 3 ).name];
                %obj.dirList = dir(obj.recDir);
            elseif strcmpi(datType,'Test')
                obj.recDir = [obj.recDir,'/',obj.dirList( mod(obj.iD,testNum) + trainNum + valdnNum + 3).name];
                %obj.dirList = dir(obj.recDir);
            end
            obj.outDir = ['./',datType];
            if ~exist(obj.outDir, 'dir')
                mkdir(obj.outDir)
            end
        end
        function loadVol(obj)
            [header_,vol_] = obj.loaddcmBCT(obj.recDir);
            obj.header = header_;
            obj.vol = vol_;
        end
        function dataGenNarrowConeAsFan(obj)
            ctr = CudaTools.Reconstruction(obj.GPUid);
            % create geometry - create CT object and do it there
            nProj = 360;
            sad = 507; % [mm]
            sdd = 700; % [mm]
            angle = linspace(0,360,nProj);
            obj.detInfo.u = 1024;
            obj.detInfo.v = 3;
            obj.detInfo.detPixSize = [0.25,0.25,1];
            u0 = 0; % [px]
            v0 = 0; % [px]
            prjs = zeros(obj.nInst, obj.detInfo.u, nProj, 'single');
            ctr.SetGeometryCircular(angle*pi/180, [0,0,1], sad, sdd, u0, v0, 0, 0, 0, 0);
            obj.pm = ctr.GetGeometry();
            if isempty(obj.nInst)
                obj.nInst = size(obj.vol,3)-20;
            end
            selectedSlice = [];
            for ijk = 1:obj.nInst
                fprintf('case: %i/%i \n',ijk,obj.nInst)
                if obj.nInst == size(obj.vol,3) -20
                    zCtr = ijk;
                    zRan = (-10:10) + zCtr + 10;
                else
                    while true
                        zCtr = randi(100);
                        zRan = (-10:10) + zCtr -50 + fix( size(obj.vol,3)/2 );
                        if isempty(find(selectedSlice == zCtr,1))
                            selectedSlice(end+1) = zCtr;
                            break
                        end
                    end
                end
                fprintf('Slicing from: %i to %i \n',zRan(1),zRan(end))
                prj = obj.vol2Prj(zRan,ctr);
                prjs(ijk,:,:) = squeeze(prj(:,2,:));
            end
            clear ctr
            g.SAD = sad; g.SDD = sdd; g.angle = angle;
            g.u = obj.detInfo.u; g.v = obj.detInfo.v;
            g.PixSize = obj.detInfo.detPixSize(1:2);
            g.UVWdim = [obj.detInfo.u,1];
            g.u0 = u0; g.v0=v0;
            g.pm = obj.pm;
            volDir = obj.recDir;

            save( sprintf('%s/2DbCT_%i_%i.mat',obj.outDir,obj.jID,obj.iD), ...
                    'prjs','g','zRan','volDir')
        end
        function prj = vol2Prj(obj,zRan,ctr)
            if nargin<3
                ctr = CudaTools.Reconstruction(obj.GPUid);
            end
            % setup volumes 
            ctr.SetImage('vol',single(obj.vol(:,:,zRan)),obj.header.PixSize);
            ctr.SetImage('lineInt',zeros(obj.detInfo.u,obj.detInfo.v,size(obj.pm,3),'single'),obj.detInfo.detPixSize);
            % project and get volume
            ctr.ProjectorSiddonForward('vol','lineInt')
            lintInt = ctr.GetImage('lineInt').values;
            prj = exp(-lintInt);
        end
    end
end