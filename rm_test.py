import os, glob
 
dir = 'static'
filelist = glob.glob(os.path.join(dir, "*"))
for f in filelist:
    os.remove(f)