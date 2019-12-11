function compute_color_inv_transform()
    t = imread('wsi_regis.png');
    [s1, s2, s3] = size(t);
    x = imread('dan_regis.png');
    x = double(reshape(x, [s1 * s2, s3])) / 255;
    t = imread('wsi_regis.png');
    t = double(reshape(t, [s1 * s2, s3])) / 255;
    x = [x.^0.5, x, x.^2, log(1.1 + x)];
    B = get_coeff(x, t);
    for i = 1:size(B, 1)
        fprintf('[');
        for j = 1:size(B, 2)
            fprintf('%.6f, ', B(i, j));
        end
        fprintf('],\n');
    end
    X = [ones(size(x, 1), 1), x];
    y = X * B;
    y = uint8(round(reshape(y, [s1, s2, s3]) * 255));
    imwrite(y, 'wsi_regis_trans.png'); end function B = get_coeff(x, t)
    X = [ones(size(x, 1), 1), x];
    B = inv(X' * X + 0.01 * eye(size(X, 2))) * X' * t; end function B = 
get_robust_coeff(x, t)
    B = zeros(size(x, 2) + 1, size(t, 2), 'double');
    for i = 1:size(t, 2)
        y = t(:, i);
        mdlr = fitlm(x, y, 'RobustOpts', 'on');
        coeff = mdlr.Coefficients.Estimate;
        B(:, i) = coeff;
    end 
end
