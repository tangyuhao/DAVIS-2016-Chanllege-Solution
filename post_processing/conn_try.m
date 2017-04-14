function conn_try( full_filename, full_savename)

    % read the images
    im1 = double(imread( full_filename ))/255;

    im1_gray = rgb2gray(im1);

    [im1_ysize, im1_xsize] = size(im1);
    
    %figure(1)
    %imshow(im1_gray);

    CC1 = bwconncomp( im1_gray ); % find connected components

    %% Image 1: find k largest connected components

    list1 = CC1.PixelIdxList; % list of connected components

    sizes1 = zeros( 1, length(list1) );
    k = min(5, length(list1));

    for i=1:length(list1);
        sizes1(i) = length( list1{i} );
    end

    [sort_sizes1, sort_idx1] = sort( sizes1, 'descend');

    select_list1 = list1( sort_idx1(1:k) ); % find the k largest islands where the elemnents are 1

    
    %figure(999);
    %hold off;
    %imshow(im1_gray);
    %hold on;

    cell_1 = cell(k,1);

    for i=1:k
        % convert to (y,x) index
        [ind_y1, ind_x1] = ind2sub( CC1.ImageSize , select_list1{i} );
        
        ind_y1 = max(1, ind_y1);
        ind_y1 = min( im1_ysize, ind_y1);
        
        ind_x1 = max(1, ind_x1);
        ind_x1 = min( im1_xsize, ind_x1);
        
        cell_1{i} = [ind_y1, ind_x1];
    end



    %% find the bounding boxes (for debugging)
    X_IND = 2;
    Y_IND = 1;

    min_x = zeros(k,1);
    min_y = zeros(k,1);

    max_x = zeros(k,1);
    max_y = zeros(k,1);

    % first 2 stores coord of bottom left corner of box, last 2 stores coord of top right corner of box
    box_coord = zeros(k,4); 

    %hold off;

    for i=1:k
         tmp_msg  = ['\leftarrow ' num2str(i)];
         tmp_text = text( double( cell_1{i}(1, X_IND) ), double( cell_1{i}(1,Y_IND) ), tmp_msg);
         tmp_text.FontSize = 15;
         tmp_text.Color = 'red';     

         min_x(i) = min( cell_1{i}(:, X_IND) );
         min_y(i) = min( cell_1{i}(:, Y_IND) );

         max_x(i) = max( cell_1{i}(:, X_IND) );
         max_y(i) = max( cell_1{i}(:, Y_IND) );

         width   = max_x(i) - min_x(i);
         height  = max_y(i) - min_y(i);

         w_epsilon = floor(width*0.15);
         h_epsilon = floor(height*0.15);

         a = max(1,min_x(i)-w_epsilon);  % min_x ++
         b = max(1,min_y(i)-h_epsilon ); % max_x ++

         c = min(width + 2*w_epsilon, CC1.ImageSize(1) );  % width ++
         d = min(height + 2*h_epsilon, CC1.ImageSize(2) ); % height ++

         box_coord(i, 1) = b;
         box_coord(i, 2) = min(b+d, CC1.ImageSize(2));
         box_coord(i, 3) = a;
         box_coord(i, 4) = min(a+c, CC1.ImageSize(1));

         %rectangle( 'Position', [a,b,c,d], 'EdgeColor', 'r' ); 
         %hold off;
    end

    %hold off;
    %hold off;


    % filtered image
    im1_filter = zeros(480,854);

    for idx=1:k
        tmp = cell_1{idx};

        for row_idx =1:size(tmp,1)
            im1_filter( tmp(row_idx,1), tmp(row_idx,2) ) = 1;
        end
    end

    %figure(777);
    %imshow(im1_filter);


    % save result
    imwrite(im1_filter, full_savename);


