function Output=parseOptions(Template,Input, varargin)
%
%    Output=parseOptions(Template,Input)
%
%Given a structure "Template" whose fields are default parameter values, overwrite
%a subset of those fields with new parameter values specified by "Inputs",
%which can be either a struct (whose fields specify new values) or a 
%cell array of string/value pairs.
%
%
%EXAMPLE: Given template and input data
% 
%                  Template.a=1;
%                  Template.b=2;
%                  Template.c=3;
% 
%                  Inputs={'a',5,'c',7};
%
%         >> Selections = parseOptions(Template,Inputs)
% 
%         Selections = 
% 
%             a: 5
%             b: 2
%             c: 7
%
%The same result can be obtained from 
%
%      InputStruct=struct(Inputs{:});
%      Selections = parseOptions(Template,InputStruct);
%
%In other words, the input parameter selections can be specified in
%structure form as well.


if isempty(Input), Input=Template; end

if iscell(Input)
    
  if length(Input)==1 && ischar(Input{1}) && strcmpi(Input,'GUI')
    
    Output = paramsdlg(Template,varargin{:});
    return
    
  elseif length(Input)==1 && isstruct(Input{1})
      
      Input=Input{1};
      
  else
      
      Input=pairs2struct(Input);

  end
end

Iflds=fieldnames(Input).';
Tflds=fieldnames(Template).';

 shn=[Tflds;Tflds];

    
if length(Template)<2

    Template(2)=struct(shn{:});
    
end

Abbrevs=Template(2);
 Template(2:end)=[];
 
 
 
 
 c=struct2cell(Abbrevs).';
  map=~(cellfun('isempty',c) | strcmp(c,Tflds));
  

 c=[shn,[c(map);Tflds(map)]];
 
 c1=c(1,:);
  if length(c1)>length(unique(c1));
     error 'Short name refers to multiple fields'
  elseif ~all(ismember(Iflds,c1))
      
      unrecognized = setdiff(Iflds,c1), %#ok<NASGU>
      
      error 'Unknown field detected'
      
  end 
 
 LookUps=struct(c{:});


 Output=Template;
 for ii=1:length(Iflds)
     
     fld=LookUps.(Iflds{ii});
     val=Input.(Iflds{ii});
     Output.(fld)=val;

 end
 


function outstruct = paramsdlg(data,varargin)
%PARAMSDLG -  Will display a dialog box allowing one to overwrite
%default field values of an input struct. This is basically a wrapper for 
%INPUTDLG meant to act something like the Variable Editor as
%applied to structs (except it will work in MATLAB Compiled standalones).
%By default the prompt string for each field will be the field name, but
%optionally, alternative prompt strings may be supplied.
%
%  outstruct = paramsdlg(defaults,dlg_title,varargin)
%
%IN:
%
% data:     A struct containing default field values. Optionally also, if
%           length(defaults)>1, then the second struct element data(2)
%           can contain alternative prompt strings for the different
%           fields. If data(2).fieldname=[], the prompt string defaults to the
%           'fieldname'. If data(2) is absent, all fields default this way.
%
% dlg_title: Title string for the dialog GUI.
%
% varargin: String-value pairs or struct, with same additional options as
%           in INPUTDLG.
%
%OUT:
%
% outstruct: New version of data(1) struct with default field values
%            overwritten as specified in GUI dialog. The entries to the
%            edit boxes in the dialog can be any single MATLAB expression
%            and can be used to specify any data type, cell, numeric,
%            struct, etc... The entries can also reference variables in the
%            base workspace. The interpretation of the entries is driven by
%            EVALIN.

%%Pre-Proc
if ~isvar('dlg_title'), dlg_title=''; end

dlgOptions.Title='';
dlgOptions.Resize='on';
dlgOptions.WindowStyle='normal';
dlgOptions.Interpreter='none';

dlgOptions(2).Title='ttl';

  dlgOptions = parseOptions(dlgOptions,varargin);

  
  
dlg_title = dlgOptions.Title;
  
 flds=fieldnames(data);
 
 nflds=length(flds);
 

    if length(data)>1
       prompts = data(2);

    else
       prompts = structfun(@(f) [], data,'uni',0);    
    end

 defaults=data(1);

 %%Format default answers for dialog display 
 dispstruct=defaults;
 for ii=1:nflds
    
     fld=flds{ii};
 
     
     if isempty(prompts.(fld))
         prompts.(fld)=fld;
     end

     val=defaults.(fld);
     
     if isnumeric(val) || iscell(val) || ischar(val) || isa(val,'function_handle')
     
         %T=mat2str(val);
         S.a=val;         
         T=evalc('disp(S)');
         T=T(find(T==':',1)+1:end);
         idx=find(T~=sprintf('\n'),1,'last');
         T(idx+1:end)='';
         T=dblnk(T);
         
     elseif isstruct(val)  
         
         C=splitstr(evalc('disp(val)'));
         C(cellfun('isempty',C))=[];
         T=char(C);
         
        %      else
        %          T=evalc('disp(val)');
        %          idx=find(T~=sprintf('\n'),1,'last');
        %          T(idx+1:end)='';
        %          T=dblnk(T);
        %          
        %          if iscell(val)
        %              T=['{ ' T ' }'];
        %          end

         
     end
     
     dispstruct.(fld)=T;

     
 end
 
 num_lines=structfun(@(s) size(s,1), dispstruct);
  num_lines(:,2)=max([num_lines;length(dlg_title)+20]);
 defAns=struct2cell(dispstruct).';
 prompts=struct2cell(prompts).';
 
 %%Launch the GUI  
 answer = inputdlg(prompts,dlg_title,num_lines,defAns,dlgOptions);
 
 
 %%Post Process output
 
 if isempty(answer), answer=defAns; end
 
 outstruct=defaults;
 for ii=1:nflds
     
     if strcmp(defAns{ii},answer{ii}), continue; end
     
       fld=flds{ii};
       str=answer{ii};
       
       if all(str([1,end])=='''') %explicit string entry

               outstruct.(fld) = str(2:end-1);
               
       else

                val=str2num(str);

                if isempty(val) && isdeployed
                     outstruct.(fld) =  eval([str, ';']);
                elseif isempty(val) && ~isdeployed   
                     outstruct.(fld) =  evalin('base', [str, ';']);
                else %non-empty val
                     outstruct.(fld) = val;
                end

        end
       
     
 end
   
 






