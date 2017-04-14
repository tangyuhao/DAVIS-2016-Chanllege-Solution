close all; clc;

post_result_path = fullfile('post_result');

% create save directory
[status, msg] = mkdir( post_result_path);

result_path = fullfile('result');


% for each subdirectory in result
yesdirs = dir( result_path);

filename = '';

if numel( yesdirs)
    % loop through folders
    for i=1:numel(yesdirs)
        
        % filter out non-directories
        if yesdirs(i).isdir && ~strcmpi(yesdirs(i).name, '.') && ~strcmpi(yesdirs(i).name, '..') 
            %disp(sprintf('%s',yesdirs(i).name));
            
            % loop through files in folder
            yesfiles = dir( fullfile('.', 'result', yesdirs(i).name) );
            
            % make folder to save file
            save_folder = fullfile('.', 'post_result', yesdirs(i).name );
            [status, msg] = mkdir( save_folder);
            
            for j=1:numel(yesfiles)
                % filter out non-directories
                if ~strcmpi(yesfiles(j).name, '.') && ~strcmpi(yesfiles(j).name, '..') 
           
                    %filename = yesfiles(j).name;
                    disp(sprintf('%s',yesfiles(j).name));
                    
                    full_filename = fullfile('.', 'result', yesdirs(i).name, yesfiles(j).name);
                    
                    % build full filename to save file
                    full_savename = fullfile('.', 'post_result', yesdirs(i).name, yesfiles(j).name);
                    %disp(sprintf('%s', full_savename));
                    
                    % execute everything !!!
                    conn_try( full_filename, full_savename);      
                end    
            end          
        end
    end    
end    
