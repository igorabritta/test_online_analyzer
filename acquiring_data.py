from root_numpy import hist2array
import ROOT
import numpy as np
import re
import time

class acquiring_data:

    def __init__(self,options):
        self.filedir = options.filedir        
        self.filename = "histograms_Run%05d.root" % options.runnumber
        self.maxImages = options.maxEntries
        self.delaytime = options.delaytime
        self.runnumber = options.runnumber

    # first calculate the mean 
    def start_acquisition(self):
        
        tf  = ROOT.TFile.Open(self.filedir+self.filename)        
        for i,e in enumerate(tf.GetListOfKeys()):
            justSkip=False
            name=e.GetName()
            obj=e.ReadObj()
            if obj.InheritsFrom('TH2'):
                obj.SetDirectory(0)
                
            if 'pic' in name:
                patt = re.compile('\S+run(\d+)_ev(\d+)')
                m = patt.match(name)
                run = int(m.group(1))
                event = int(m.group(2))
            if self.maxImages>-1 and event>min(len(tf.GetListOfKeys()),self.maxImages): break

            if not obj.InheritsFrom('TH2'): justSkip=True
            
            if justSkip:
                obj.Delete()
                del obj
                continue
                    
            arr = hist2array(obj)
            
            ## Solve memory leak
            obj.Delete()
            del obj
            
            np.save("arriving/run%05d_image%04d" % (self.runnumber,i), arr)
            print("Run %05d Image %04d saved successfully " % (self.runnumber,i))
            time.sleep(self.delaytime)
        tf.Close()

if __name__ == '__main__':
    from optparse import OptionParser
    
    ## python3 acquiring_data.py -r 6332 --max-entries 40 -d 7
    
    parser = OptionParser(usage='%prog h5file1,...,h5fileN [opts] ')
    parser.add_option('-r', '--run', dest='runnumber', default='00000', type='int', help='run number with 5 characteres')
    parser.add_option(      '--max-entries', dest='maxEntries', default=-1, type='int', help='Process only the first n entries')
    parser.add_option(      '--dir', dest='filedir', default='/jupyter-workspace/cloud-storage/cygno-data/LAB/', type='string', help='Directory where the input files are')
    parser.add_option('-d','--delay', dest='delaytime', default=5, type='int', help='delay time to save the images')
    
    (options, args) = parser.parse_args()
    
    ad = acquiring_data(options)
    ad.start_acquisition()
    