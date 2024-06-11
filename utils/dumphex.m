function out =  dumphex(filename, n)
% read 16*n bytes
% ex. dumphex('picture.bmp', 4)

fid=fopen(filename, 'r');

if fid == -1
    error('File does not exist');
end

a=fread(fid, 16*n, 'uchar');
idx=find(a>=32 & a<=126);
ah=dec2hex(a);
b=repmat(' ', 16*n, 3);
b2=repmat('.', 16, n);
b2(idx)=char(a(idx));
b(:,1:2)=ah;

out = [reshape(b',48,n)' repmat(' ',n,2) reshape(b2,16,n)'];
end