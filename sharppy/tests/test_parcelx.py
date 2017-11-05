import sharppy.sharptab.params as params
import sharppy.sharptab.profile as profile
import sharppy.io.spc_decoder as decoder
import glob
import numpy as np
from pylab import *
from datetime import datetime
from StringIO import StringIO

def parseSPC(spc_file):
    ## read in the file
    data = np.array([l.strip() for l in spc_file.split('\n')])

    ## necessary index points
    title_idx = np.where( data == '%TITLE%')[0][0]
    start_idx = np.where( data == '%RAW%' )[0] + 1
    finish_idx = np.where( data == '%END%')[0]

    ## create the plot title
    data_header = data[title_idx + 1].split()
    location = data_header[0]
    time = data_header[1][:11]

    ## put it all together for StringIO
    full_data = '\n'.join(data[start_idx : finish_idx][:])
    sound_data = StringIO( full_data )

    ## read the data into arrays
    p, h, T, Td, wdir, wspd = np.genfromtxt( sound_data, delimiter=',', comments="%", unpack=True )

    return p, h, T, Td, wdir, wspd

sars_files = np.sort(glob.glob('../databases/sars/hail/*'))

for flag in [1,3,4]:
    time = []
    cape = []
    cin = []
    lcl = []
    lfc = []
    el = []
    
    for f in sars_files[:150]:
        print "File:", f
        spc_file = open(f, 'r').read()
        pres, hght, tmpc, dwpc, wdir, wspd = parseSPC(spc_file)

        dt = datetime.now()    
        prof = profile.create_profile(profile='default', pres=pres, hght=hght, tmpc=tmpc, \
                                        dwpc=dwpc, wspd=wspd, wdir=wdir, missing=-9999, strictQC=False)
        delta_time = datetime.now() - dt
        print "Time to make profile object:", delta_time

        print "\nTest WOBUS:"
        dt = datetime.now()
        pcl1 = params.parcelx(prof, method='wobus', flag=flag)
        tfl1 = datetime.now() - dt
        print pcl1.bplus, pcl1.bminus, pcl1.lclhght, pcl1.lfchght, pcl1.elhght, pcl1.mplhght
        print "Time for Lift:",tfl1

        print "\nTest RDJ:"
        dt = datetime.now()
        pcl2 = params.parcelx(prof, method='bolton', flag=flag)
        tfl2 = datetime.now() - dt
        print pcl2.bplus, pcl2.bminus, pcl2.lclhght, pcl2.lfchght, pcl2.elhght, pcl2.mplhght
        print "Time for Lift:", tfl2
        cape.append([pcl1.bplus, pcl2.bplus])
        cin.append([pcl1.bminus, pcl2.bminus])
        lcl.append([pcl1.lclhght, pcl2.lclhght])
        lfc.append([pcl1.lfchght, pcl2.lfchght])
        el.append([pcl1.elhght, pcl2.elhght])
        time.append([tfl1.total_seconds(), tfl2.total_seconds()])
        print "\n\n"
    #stop
    print lfc
    cape = np.asarray(cape)
    cin = np.asarray(cin)
    lcl = np.asarray(lcl)
    lfc = np.ma.asarray(lfc)
    el = np.ma.asarray(el)
    time = np.asarray(time)

    for index in [time, cape, cin, lcl, lfc, el]:
        index = np.ma.masked_invalid(index)
        print index[:,0], index[:,1]
        plot(index[:,0], index[:,1], 'o')
        xlim(np.ma.min(index),np.ma.max(index))
        ylim(np.ma.min(index),np.ma.max(index))
        plot([np.ma.min(index), np.ma.max(index)], [np.ma.min(index), np.ma.max(index)], 'k-')
        xlabel('WOBUS')
        grid()
        ylabel("RDJ")
        show()
