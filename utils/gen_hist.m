function ret = gen_hist(I)
    % Initial histogram vector
    ret = zeros(256, 1) ;
    [row, col] = size(I);

    for x = 1:row
        for y = 1:col
            ret(I(x, y) + 1) = ret(I(x, y) + 1) + 1;
        end
    end
end