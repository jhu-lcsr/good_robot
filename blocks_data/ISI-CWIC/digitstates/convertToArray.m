imagefiles = dir('images/*.png');
nfiles = length(imagefiles);
nfiles = length(imagefiles);
fid=fopen('images_ids.txt', 'wt');
for ii=1:nfiles
  currentfilename = imagefiles(ii).name;
  currentimage = rgb2gray(imread(['images/',currentfilename]));
  fprintf(fid, '%s\n',currentfilename);
  dlmwrite('images_arrays.txt',currentimage(:)','-append');
end
fclose(fid);
