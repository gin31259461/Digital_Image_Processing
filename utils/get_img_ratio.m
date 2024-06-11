function ratio = get_img_ratio(filename)
K = imfinfo(filename);
ratio = (K.Width * K.Height * K.BitDepth/8) / K.FileSize;
end