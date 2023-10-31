classdef tqdm < handle
% class for command-line progress-bar notification in tqdm style.
% Use example:
%   pb = tqdm('Doing stuff...');
%   for k = 1 : 10
%       pb.print(k,10)
%       % do stuff
%   end
%
% Author: Heyuan Huang

% Modification 07.23.2023
% Added option to specific total number of items when initialize
% would not break previous workflow

    properties
        last_msg_len = 0;
        firstItem = true;
        startTime
        totNum = inf;
    end
    methods
        %--- ctor
        function obj = tqdm(msg,tot)
%             obj.startTime = tic;
            fprintf('%s  ', msg)
            if exist('tot','var')
                obj.totNum = tot;
            end
        end
        %--- print method
        function print(obj, n, tot)
            if exist('tot','var')
                obj.totNum = tot;
            end
            if obj.firstItem
                obj.firstItem = false;
                obj.startTime = tic;
            end
            fprintf('%s', char(8*ones(1, obj.last_msg_len))) % delete last info_str
            elapsedTime = toc(obj.startTime);
            secPerItem = elapsedTime/n;
            ETF = (obj.totNum-n)*secPerItem;
            info_str = sprintf('%d%%, %d/%d, %02i:%02i <-- %02i:%02i, %.1f sec/item',...
                                fix(n/obj.totNum*100),n,obj.totNum,...
                                fix(elapsedTime/60),fix(mod(elapsedTime,60)),...
                                fix(ETF/60),fix(mod(ETF,60)),secPerItem);
            fprintf('%s', info_str);
            %--- assume user counts monotonically
            if n == obj.totNum
                fprintf('\n')
            end
            obj.last_msg_len = length(info_str);
        end
        %--- dtor
        function delete(obj)
            fprintf('\n')
        end
    end
end