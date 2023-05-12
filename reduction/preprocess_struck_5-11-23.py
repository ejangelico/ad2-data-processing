import sys
import os
sys.path.append("../Dataset/")
sys.path.append("../ParseStruck/")
import NGMBinaryFile
import glob





if __name__ == "__main__":

    if(len(sys.argv) != 3):
        print("python preprocess_struck_<>.py <path to data> <path to channel map>")
        sys.exit()


    topdir = sys.argv[1]
    chmap = sys.argv[2]
    
    infiles = glob.glob(topdir+"*.bin")
    print(infiles)
    for f in infiles:
        ngmb = NGMBinaryFile.NGMBinaryFile(input_filename=f, output_directory=topdir, channel_map_file = chmap)
        ngmb.GroupEventsAndWriteToPickle(save=True)
        break


    
