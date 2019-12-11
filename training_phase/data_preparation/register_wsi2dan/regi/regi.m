function register_wsi()
    register('dan.png', 'wsi.png', 'dan_regis.png', 'wsi_regis.png', 
'similarity', 4);
    register('dan_regis.png', 'wsi_regis.png', 'dan_regis.png', 
'wsi_regis.png', 'similarity', 2);
    register('dan_regis.png', 'wsi_regis.png', 'dan_regis.png', 
'wsi_regis.png', 'affine', 2); end function register(fix, move, fix_out, 
move_out, method, shrink)
    dan = imread(fix);
    wsi = imread(move);
    [optimizer, metric] = imregconfig('multimodal');
    tform = imregtform(rgb2gray(wsi), rgb2gray(dan), method, optimizer, 
metric);
    wsi_regis = imwarp(wsi, tform, 'OutputView', imref2d(size(dan)));
    mask = (wsi_regis > 0 & dan > 0);
    mask = mask(:, :, 1);
    mask = bwmorph(mask, 'shrink', shrink);
    wsi_regis(:, :, 1) = wsi_regis(:, :, 1) .* uint8(mask > 0);
    wsi_regis(:, :, 2) = wsi_regis(:, :, 2) .* uint8(mask > 0);
    wsi_regis(:, :, 3) = wsi_regis(:, :, 3) .* uint8(mask > 0);
    dan(:, :, 1) = dan(:, :, 1) .* uint8(mask > 0);
    dan(:, :, 2) = dan(:, :, 2) .* uint8(mask > 0);
    dan(:, :, 3) = dan(:, :, 3) .* uint8(mask > 0);
    imwrite(dan, fix_out);
    imwrite(wsi_regis, move_out); end
