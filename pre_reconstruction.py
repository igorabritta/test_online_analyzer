import numpy as np
import ROOT
import time
import toolslib as tl

from root_numpy import hist2array
from sklearn.cluster import DBSCAN
from cameraChannel import cameraTools as ctools
from scipy.ndimage import gaussian_filter, median_filter
from clusterTools import Cluster

class Variables:
    def __init__(self,runnumber,i):
        self.runnumber = runnumber
        self.nev       = i

def pre_reconstruction(arr,runnumber,i,pedmap,pedsigma,printtime=False):
    t_ini = time.time()

    ################ analysis cards ################################
    nsigma       = 1.4         # numero di sigma sopra il piedistallo
    cimax        = 5000       # valori del cut sull'imagine
    rebin        = 4       # binnagio finale immagine (deve essre un sottomultipli della 2**2 risluzione di partenza)
    eps          = 2.2         # maximum distance for the cluster point
    min_samples  = 50
    ymin         = 200
    ymax         = 2304-100
    xmin         = 0
    xmax         = 2304
    npixx        = 2304
    self         = []
    
    pedarr_fr    = pedmap
    sigarr_fr    = pedsigma
    img_fr = arr
    
    #################################
    # Preprocessing steps
    ##################################

    # Upper Threshold full image
    img_cimax = np.where(img_fr < cimax, img_fr, 0)

    img_fr_sub = ctools.pedsub(self,img_cimax,pedarr_fr)
    img_fr_satcor = img_fr_sub  
    img_fr_zs  = ctools.zsfullres(self,img_fr_satcor,sigarr_fr,nsigma=nsigma)
    img_fr_zs_acc = ctools.acceptance(self,img_fr_zs,ymin,ymax,xmin,xmax)
    img_rb_zs  = ctools.arrrebin(self,img_fr_zs_acc,rebin,npixx)

    th_image = pedarr_fr

    #-----Pre-Processing----------------#
    rescale=int(npixx/rebin)

    filtimage = median_filter(img_fr_zs, size=2)
    edges = ctools.arrrebin(self,filtimage,rebin,npixx)
    edcopy = edges.copy()
    edcopyTight = edcopy
    #edcopyTight = tl.noisereductor(edcopy,rescale,self.options.min_neighbors_average)

    # make the clustering with DBSCAN algo
    # this kills all macrobins with N photons < 1
    points = np.array(np.nonzero(np.round(edcopyTight))).astype(int).T
    lp = points.shape[0]

    sample_weight = np.take(img_rb_zs, img_rb_zs.shape[0]*points[:,0]+points[:,1]).astype(int)
    sample_weight[sample_weight==0] = 1
    X = points.copy()

    t0 = time.time()
    ddb = DBSCAN(eps=eps,min_samples=min_samples, metric='cityblock').fit(X,sample_weight = sample_weight)
    t1 = time.time()


    # Black removed and is used for noise instead.
    unique_labels = set(ddb.labels_)

    # Number of polynomial clusters in labels, ignoring noise if present.
    n_superclusters = len(unique_labels) - (1 if -1 in ddb.labels_ else 0)
    # returned collections
    superclusters = []

    t2 = time.time()        
    for k in unique_labels:
        if k == -1:
            break # noise: the unclustered

        class_member_mask = (ddb.labels_ == k)
        #class_member_mask = (ddb.labels_ == k)
        xy = np.unique(X[class_member_mask],axis=0)
        x = xy[:, 0]; y = xy[:, 1]


        # both core and neighbor samples are saved in the cluster in the event
        if k>-1 and len(x)>1:
            cl = Cluster(xy,rebin,img_fr,img_fr_zs,'lime',debug=False,fullinfo=False,clID=k)
            cl.iteration = 0
            superclusters.append(cl)


    for k,cl1 in enumerate(superclusters):
        cl1.calcShapes()
    t3 = time.time()

    ## Calculating variables

    t4 = time.time()
    variables = Variables(runnumber,i)
    variables.cl_size     = [cl.size() for cl in superclusters]
    variables.cl_nhits    = [cl.sizeActive() for cl in superclusters]
    variables.cl_integral = [cl.integral() for cl in superclusters]

    #cl_length   = [cl.shapes['long_width'] for cl in superclusters]
    #cl_width    = [cl.shapes['lat_width'] for cl in superclusters]

    variables.cl_xmean    = [cl.shapes['xmean'] for cl in superclusters]
    variables.cl_ymean    = [cl.shapes['ymean'] for cl in superclusters]
    variables.cl_xmax     = [cl.shapes['xmax'] for cl in superclusters]
    variables.cl_xmin     = [cl.shapes['xmin'] for cl in superclusters]
    variables.cl_ymax     = [cl.shapes['ymax'] for cl in superclusters]
    variables.cl_ymin     = [cl.shapes['ymin'] for cl in superclusters]

    variables.run         = (runnumber*(np.ones((1,len(variables.cl_integral)),dtype=int))[0]).tolist()
    variables.event       = (i*(np.ones((1,len(variables.cl_integral)),dtype=int))[0]).tolist()
    variables.nclu        = list(range(1,len(variables.cl_integral)))

    t5 = time.time()

    t_tot = time.time()
    if printtime:
        print ("Step 1: Clusterization: %.2f seconds / %.2f minutes" % ((t1-t0),(t1-t0)/60))
        print ("Step 2: Cluster information: %.2f seconds / %.2f minutes" % ((t3-t2),(t3-t2)/60))
        print ("Step 3: Variables calculation: %.2f seconds / %.2f minutes" % ((t5-t4),(t5-t4)/60))
        print ("Total Elapsed time: %.2f seconds / %.2f minutes" % ((t_tot-t_ini),(t_tot-t_ini)/60))
    
    return variables

def savedata_totable(df,variables):
    # Import pandas library
    import pandas as pd

    # initialize list of lists
    data = [variables.run, variables.event, variables.nclu, variables.cl_size, variables.cl_nhits, variables.cl_integral, variables.cl_xmean, variables.cl_ymean, variables.cl_xmax, variables.cl_xmin, variables.cl_ymax, variables.cl_ymin]
    data = list(map(list, zip(*data)))
    
    if len(df) == 0:
        # Create the pandas DataFrame
        df = pd.DataFrame(data, columns = ['Run','Event','Nclu','Size','nhits','Integral','xmean','ymean','xmax','xmin','ymax','ymin'])
    else:
        df_new = pd.DataFrame(data, columns = ['Run','Event','Nclu','Size','nhits','Integral','xmean','ymean','xmax','xmin','ymax','ymin'])
        df = df.append(df_new)

    # save dataframe.
    df.to_csv('raw_data.csv', index=False)

def opendata_table(filename):
    import pandas as pd
    try:
        df = pd.read_csv(filename)
        print("Found Table")
    except:
        print("Table not found")
        df = []
    return df