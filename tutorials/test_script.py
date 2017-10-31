# Attempting to both accelerate and make parcelx() from SHARPpy more flexible.
import skewt
import sharppy.sharptab.params as params
import sharppy.sharptab.thermo as thermo
import sharppy.sharptab.interp as interp
import sharppy.sharptab.profile as profile
import sharppy.sharptab.utils as utils
import numpy as np
import numpy.ma as ma
from StringIO import StringIO

G = 9.81 

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

def parcelx(prof, pbot=None, ptop=None, dp=-1, method='bolton', **kwargs):
    '''
        Lifts the specified parcel, calculated various levels and parameters from
        the profile object. B+/B- are calculated based on the specified layer.
        
        !! All calculations use the virtual temperature correction unless noted. !!
        
        Parameters
        ----------
        prof : profile object
        Profile Object
        pbot : number (optional; default surface)
        Pressure of the bottom level (hPa)
        ptop : number (optional; default 400 hPa)
        Pressure of the top level (hPa)
        pres : number (optional)
        Pressure of parcel to lift (hPa)
        tmpc : number (optional)
        Temperature of parcel to lift (C)
        dwpc : number (optional)
        Dew Point of parcel to lift (C)
        dp : negative integer (optional; default = -1)
        The pressure increment for the interpolated sounding
        exact : bool (optional; default = False)
        Switch to choose between using the exact data (slower) or using
        interpolated sounding at 'dp' pressure levels (faster)
        flag : number (optional; default = 5)
        Flag to determine what kind of parcel to create; See DefineParcel for
        flag values
        lplvals : lifting parcel layer object (optional)
        Contains the necessary parameters to describe a lifting parcel
        
        Returns
        -------
        pcl : parcel object
        Parcel Object
        
        '''
    flag = kwargs.get('flag', 5)
    pcl = params.Parcel(pbot=pbot, ptop=ptop) # Create an empty parcel
    pcl.lplvals = kwargs.get('lplvals', params.DefineParcel(prof, flag)) # Set the original parcel values
    if prof.pres.compressed().shape[0] < 1: return pcl
    
    # Variables
    pres = kwargs.get('pres', pcl.lplvals.pres)
    tmpc = kwargs.get('tmpc', pcl.lplvals.tmpc)
    dwpc = kwargs.get('dwpc', pcl.lplvals.dwpc)
    pcl.pres = pres
    pcl.tmpc = tmpc
    pcl.dwpc = dwpc
    pcl.thetae = thermo.thetae(pcl.pres, pcl.tmpc, pcl.dwpc, method=method)
    cap_strength = -9999.
    cap_strengthpres = -9999.
    li_max = -9999. # maximum LI
    li_maxpres = -9999. # pressure of the maximum LI
    totp = 0. # total positive energy
    totn = 0. # total negative energy
    tote = 0. # total energy
    cinh_old = 0.
    
    # See if default layer to lift over is specificed
    if not pbot: # User didn't specify the bottom of the layer.
        pbot = prof.pres[prof.sfc]
        pcl.blayer = pbot
        pcl.pbot = pbot
        # Set it to be the pressure at the bottom of the sounding .
    if not ptop: # User didn't specify the bottom of the layer.
        ptop = prof.pres[prof.pres.shape[0]-1]
        pcl.tlayer = ptop
        pcl.ptop = ptop
        # Set it to be the pressure at the bottom of the sounding .
    # Make sure this is a valid layer
    if pbot > pres: # if the specified pbot is greater than LPLPRES (e.g., elevated MU parcel)
        pbot = pres
        pcl.blayer = pbot
        # set the bottom of the layer we're lifting over to LPLPRES.
        
    # if the virt. temp. at the top and bottom is masked, return a masked value.
    if type(interp.vtmp(prof, pbot)) == type(ma.masked): return ma.masked
    if type(interp.vtmp(prof, ptop)) == type(ma.masked): return ma.masked
    
    # Nomenclature:
    # pe1 - [p]ressure [e]nvironment 1
    # te1 - virtual [t]emperature [e]nvironment 1
    # tp1 - virtual [t]emperature [p]arcel 1
    
    # Begin with the bottom of the layer
    pe1 = pbot
    h1 = interp.hght(prof, pe1)
    tp1 = thermo.virtemp(pres, tmpc, dwpc)
    
    # Lift parcel and return LCL pres (hPa) and LCL temp (C)
    pe2, tp2 = thermo.drylift(pres, tmpc, dwpc, method=method)
    blupper = pe2 # pressure at the LCL
    h2 = interp.hght(prof, pe2)
    te2 = interp.vtmp(prof, pe2)
    
    # Store the LCL values to the Parcel object.
    pcl.lclpres = min(pe2, prof.pres[prof.sfc]) # Make sure the LCL pressure is
                                                # never below the surface
    pcl.lclhght = interp.to_agl(prof, h2)
    
    # Append the p & T_v LPL and LCL points to the parcel trace arrays
    ptrace = np.asarray([pe1,pe2])
    ttrace  = np.asarray([tp1,thermo.virtemp(pe2, tp2, tp2)]) # Keep this profile in virtual temperature space
    
    # Calculate lifted parcel theta for use in iterative CINH loop below
    # RECALL: lifted parcel theta is CONSTANT from LPL to LCL
    theta_parcel = thermo.theta(pe2, tp2, 1000.) # This is theta for parcel at LCL == theta at LPL.
    
    # Environmental theta and mixing ratio at LPL
    bltheta = thermo.theta(pres, interp.temp(prof, pres), 1000.) #Calculate the theta of the parcel at the LPL
    blmr = thermo.mixratio(pres, dwpc) # Calculate the WVMR of the parcel.
    
    # ACCUMULATED CINH IN THE MIXED LAYER BELOW THE LCL
    # This will be done in 'dp' increments and will use the virtual
    # temperature correction where possible
    pp = np.arange(pbot, blupper+dp, dp, dtype=type(pbot)) # Get pressure at dp increments from bottom of layer to LCL
    hh = interp.hght(prof, pp) # Get the heights from those pressures
    tmp_env_theta = thermo.theta(pp, interp.temp(prof, pp), 1000.) # Get thetas over that layer.
    tmp_env_dwpt = interp.dwpt(prof, pp) # Calculate the dewpoints over that layer.
    tv_env = thermo.virtemp(pp, tmp_env_theta, tmp_env_dwpt) # Calculate the environ. virt. temperature over the layer.
    tmp1 = thermo.virtemp(pp, theta_parcel, thermo.temp_at_mixrat(blmr, pp)) # calculate Theta_v throughout the pp layer.
    tdef = (tmp1 - tv_env) / thermo.ctok(tv_env) # (T_p - T_e)/(T_e)

    tidx1 = np.arange(0, len(tdef)-1, 1)
    tidx2 = np.arange(1, len(tdef), 1)
    #print tidx1, tidx2
    lyre = G * (tdef[tidx1]+tdef[tidx2]) / 2 * (hh[tidx2]-hh[tidx1])
    #print lyre
    totn = lyre[lyre < 0].sum()
    if not totn: totn = 0.
    
    # Move the bottom layer to the top of the boundary layer/LCL
    if pbot > pe2:
        pbot = pe2
        pcl.blayer = pbot
    
    # Calculate height of various temperature levels
    p0c = params.temp_lvl(prof, 0.)
    pm10c = params.temp_lvl(prof, -10.)
    pm20c = params.temp_lvl(prof, -20.)
    pm30c = params.temp_lvl(prof, -30.)
    hgt0c = interp.hght(prof, p0c)
    hgtm10c = interp.hght(prof, pm10c)
    hgtm20c = interp.hght(prof, pm20c)
    hgtm30c = interp.hght(prof, pm30c)
    pcl.p0c = p0c
    pcl.pm10c = pm10c
    pcl.pm20c = pm20c
    pcl.pm30c = pm30c
    pcl.hght0c = hgt0c
    pcl.hghtm10c = hgtm10c
    pcl.hghtm20c = hgtm20c
    pcl.hghtm30c = hgtm30c

    if pbot < prof.pres[-1]:
        # Check for the case where the LCL is above the 
        # upper boundary of the data (e.g. a dropsonde)
        return pcl

    # Find indices corresponding to the top and bottom of the new layer that is above the LCL.
    lptr = ma.where(pbot >= prof.pres)[0].min()
    uptr = ma.where(ptop <= prof.pres)[0].max()
    
    # START WITH INTERPOLATED BOTTOM LAYER
    # Begin moist ascent from lifted parcel LCL (pe2, tp2)
    pe1 = pbot
    h1 = interp.hght(prof, pe1)
    te1 = interp.vtmp(prof, pe1)
    tp1 = tp2
    lyre = 0 # layer energy
    lyrlast = 0 # energy of the last layer considered
    
    iter_ranges = np.arange(lptr, prof.pres.shape[0])
    #ttraces = ma.zeros(len(iter_ranges))
    #ptraces = ma.zeros(len(iter_ranges))
    pcl.thetae = thermo.thetae(ptrace[1],  tp2, tp2)
    #print ptrace[1], tp2, pcl.thetae
    #pcl.thetae = thermo.thetae(ptrace[0],  ttrace[0], ttrace[0])
    #print ptrace[0], ttrace[0], pcl.thetae
    ttraces = thermo.wetlift(1000,25, prof.pres[iter_ranges], theta_e=pcl.thetae, method='bolton')
    ptraces = prof.pres[iter_ranges] # Obtain the pressure grid of the pseudoadiabatic portion of the profile
    tv_env = prof.vtmp[iter_ranges] # The environmental virtual temperature profile
    tv_pcl = thermo.virtemp(ptraces, ttraces, ttraces) # Covert the parcel temperature into virtual temperature space
    B = (tv_pcl - tv_env) / thermo.ctok(tv_env) # buoyancy profile along the pseudoadiabat
    #print ptraces, tv_pcl, ttraces
    #stop
    #print "i     pe2    tp2     pe1    te2   tp1  lyre lyrlast tote totp   totn"      
    for i in iter_ranges:
        if not utils.QC(prof.tmpc[i]): continue # Continue if the temperature value given is invalid
        pe2 = ptraces[i-iter_ranges[0]]
        h2 = prof.hght[i]
        te2 = tv_env[i-iter_ranges[0]]
        tp2 = ttraces[i-iter_ranges[0]]# thermo.wetlift(pe1, tp1, pe2) # lift the parcel to the next level.
        tdef1 = (thermo.virtemp(pe1, tp1, tp1) - te1) / thermo.ctok(te1) # normalized buoyancy at the bottom of the layer
        tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / thermo.ctok(te2) # normalized buoyancy at the top of the layer
        #ttraces[i-iter_ranges[0]] = thermo.virtemp(pe2, tp2, tp2)
        # Add the parcel values to the array keeping track of the parcel trace.
        lyrlast = lyre # Save the layer energy to the last layer energy...this will be used to find levels where the energy sign reverses (e.g. LFC)
        lyre = G * (tdef1 + tdef2) / 2. * (h2 - h1) # compute the layer energy (J/kg)

        # Add layer energy to total positive if lyre > 0
        if lyre > 0: totp += lyre
        # Add layer energy to total negative if lyre < 0, only up to 500 mb.
        else:
            if pe2 > 500.: totn += lyre
        #print "%d  %1.2f  %1.2f  %1.2f  %1.2f  %1.2f  %1.2f  %1.2f  %1.2f  %1.2f  %1.2f" % (i-iter_ranges[0], pe2, thermo.virtemp(pe2, tp2, tp2), pe1, te2, thermo.virtemp(pe1, tp1, tp1), lyre, lyrlast, tote, totp, totn)      
        
        # Check for Max LI
        mli = tv_pcl[i-iter_ranges[0]] - te2
        if  mli > li_max:
            li_max = mli
            li_maxpres = pe2
        
        # Check for Max Cap Strength
        mcap = te2 - mli
        if mcap > cap_strength:
            cap_strength = mcap
            cap_strengthpres = pe2
        
        tote += lyre # Add the current layer's energy to the total energy
        pelast = pe1 # Store the pressure at the bottom of the layer.
        # Set the values at the top of the considered layer to the values as the bottom vals. of the next iteration's layer
        pe1 = pe2 
        te1 = te2
        tp1 = tp2
        
        # Is this the top of the specified layer we're lifting the parcel over
        if i >= uptr and not utils.QC(pcl.bplus):
            #print "Found the top of the specified layer we're lifting over."
            pe3 = pe1
            h3 = h2
            te3 = te1
            tp3 = tp1
            lyrf = lyre
            if lyrf > 0:
                pcl.bplus = totp - lyrf
                pcl.bminus = totn
            else:
                pcl.bplus = totp
                if pe2 > 500.: pcl.bminus = totn + lyrf
                else: pcl.bminus = totn
            #pe2 = ptop
            #h2 = interp.hght(prof, pe2)
            #te2 = interp.vtmp(prof, pe2)
            #tp2 = thermo.wetlift(pe3, tp3, pe2)
            #tp2 = tp3
            #print "\tLifting from pe3 to pe2:", pe3, te3, pe2
            #tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / thermo.ctok(te3)
            #tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / thermo.ctok(te2)
            #lyrf = G * (tdef3 + tdef2) / 2. * (h2 - h3)
            #print "\tlyrf:", lyrf
            lyrf = 0
            if lyrf > 0: pcl.bplus += lyrf
            else:
                if pe2 > 500.: pcl.bminus += lyrf
            if pcl.bplus == 0: pcl.bminus = 0.
        
        # Is this the freezing level
        if te2 < 0. and not utils.QC(pcl.bfzl):
            pe3 = pelast = ptraces[i-iter_ranges[0]-1]
            h3 = interp.hght(prof, pe3)
            te3 = tv_env[i-iter_ranges[0]-1]
            tp3 = tv_pcl[i-iter_ranges[0]-1] # in virtual tempreature space
            lyrf = lyre
            if lyrf > 0.: pcl.bfzl = totp - lyrf
            else: pcl.bfzl = totp
            #print "\tpcl.bfzl:", pcl.bfzl
            if not utils.QC(p0c) or p0c > pe3:
                pcl.bfzl = 0
            elif utils.QC(pe2):
                te2 = tv_env[i-iter_ranges[0]]
                tp2 = tv_pcl[i-iter_ranges[0]]
                tdef3 = (tp3 - te3) / thermo.ctok(te3)
                tdef2 = (tp2 - te2) / thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (hgt0c - h3)
                if lyrf > 0: pcl.bfzl += lyrf
            #print "\tpcl.bfzl:", pcl.bfzl
        
        # Is this the -10C level
        if te2 < -10. and not utils.QC(pcl.wm10c):
            pe3 = pelast = ptraces[i-iter_ranges[0]-1]
            h3 = interp.hght(prof, pe3)
            te3 = tv_env[i-iter_ranges[0]-1]
            tp3 = tv_pcl[i-iter_ranges[0]-1] # in virtual tempreature space
            lyrf = lyre
            if lyrf > 0.: pcl.wm10c = totp - lyrf
            else: pcl.wm10c = totp
            if not utils.QC(pm10c) or pm10c > pcl.lclpres:
                pcl.wm10c = 0
            elif utils.QC(pe2):
                te2 = tv_env[i-iter_ranges[0]]
                tp2 = tv_pcl[i-iter_ranges[0]]
                tdef3 = (tp3 - te3) / thermo.ctok(te3)
                tdef2 = (tp2 - te2) / thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (hgtm10c - h3) # energy between pe3 and pe2
                if lyrf > 0: pcl.wm10c += lyrf
        
        # Is this the -20C level
        if te2 < -20. and not utils.QC(pcl.wm20c):
            pe3 = pelast = ptraces[i-iter_ranges[0]-1]
            h3 = interp.hght(prof, pe3)
            te3 = tv_env[i-iter_ranges[0]-1]
            tp3 = tv_pcl[i-iter_ranges[0]-1] # in virtual tempreature space
            lyrf = lyre
            if lyrf > 0.: pcl.wm20c = totp - lyrf
            else: pcl.wm20c = totp
            if not utils.QC(pm20c) or pm20c > pcl.lclpres:
                pcl.wm20c = 0
            elif utils.QC(pe2):
                te2 = tv_env[i-iter_ranges[0]]
                tp2 = tv_pcl[i-iter_ranges[0]]
                tdef3 = (tp3 - te3) / thermo.ctok(te3)
                tdef2 = (tp2 - te2) / thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (hgtm20c - h3)
                if lyrf > 0: pcl.wm20c += lyrf
        
        # Is this the -30C level
        if te2 < -30. and not utils.QC(pcl.wm30c):
            pe3 = pelast = ptraces[i-iter_ranges[0]-1]
            h3 = interp.hght(prof, pe3)
            te3 = tv_env[i-iter_ranges[0]-1]
            tp3 = tv_pcl[i-iter_ranges[0]-1] # in virtual tempreature space
            lyrf = lyre
            if lyrf > 0.: pcl.wm30c = totp - lyrf
            else: pcl.wm30c = totp
            if not utils.QC(pm30c) or pm30c > pcl.lclpres:
                pcl.wm30c = 0
            elif utils.QC(pe2):
                te2 = tv_env[i-iter_ranges[0]]
                tp2 = tv_pcl[i-iter_ranges[0]]
                tdef3 = (tp3 - te3) / thermo.ctok(te3)
                tdef2 = (tp2 - te2) / thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (hgtm30c - h3)
                if lyrf > 0: pcl.wm30c += lyrf
        
        # Does the parcel saturate below 3000 m?
        if pcl.lclhght < 3000.:
            # Is the 3000 m level somewhere between the top (2) and bottom (1) of the layer?
            if interp.to_agl(prof, h1) <=3000. and interp.to_agl(prof, h2) >= 3000. and not utils.QC(pcl.b3km):
                #print "Found the 3000 m level."
                pe3 = pelast = ptraces[i-iter_ranges[0]-1]
                h3 = interp.hght(prof, pe3)
                te3 = tv_env[i-iter_ranges[0]-1]
                tp3 = tv_pcl[i-iter_ranges[0]-1]
                lyrf = lyre
                #print "\tLifting to pelast:", pe3, te3, tp3, lyrf
                if lyrf > 0: pcl.b3km = totp - lyrf
                else: pcl.b3km = totp
                h4 = interp.to_msl(prof, 3000.)
                pe4 = interp.pres(prof, h4)
                #print "\tpe4:", h4, pe4
                if utils.QC(pe2):
                    #print '\tLifting to 3000 m AGL from pe3:'
                    te2 = interp.vtmp(prof, pe4)
                    tp2 = thermo.wetlift(pe3, tp3, pe4, pcl.thetae, method=method)[0]
                    tdef3 = (tp3 - te3) / thermo.ctok(te3)
                    tdef2 = (thermo.virtemp(pe4, tp2, tp2) - te2) / \
                        thermo.ctok(te2)
                    lyrf = G * (tdef3 + tdef2) / 2. * (h4 - h3)
                    #print '\tEnergy over this layer:', lyrf
                    if lyrf > 0: pcl.b3km += lyrf
                #print "\tb3km:", pcl.b3km
        else: pcl.b3km = 0.
         
        # Does the parcel saturate below 6000 m?
        if pcl.lclhght < 6000.:
            # Is the 6000 m level somewhere between the top (2) and bottom (1) of the layer?
            if interp.to_agl(prof, h1) <=6000. and interp.to_agl(prof, h2) >= 6000. and not utils.QC(pcl.b6km):
                #print "Found the 6000 m level."
                pe3 = pelast = ptraces[i-iter_ranges[0]-1]
                h3 = interp.hght(prof, pe3)
                te3 = tv_env[i-iter_ranges[0]-1]
                tp3 = tv_pcl[i-iter_ranges[0]-1]
                lyrf = lyre
                if lyrf > 0: pcl.b6km = totp - lyrf
                else: pcl.b6km = totp
                h4 = interp.to_msl(prof, 6000.)
                pe4 = interp.pres(prof, h4)
                if utils.QC(pe2):
                    te2 = interp.vtmp(prof, pe4)
                    tp2 = thermo.wetlift(pe3, tp3, pe4, pcl.thetae, method=method)[0]
                    tdef3 = (tp3 - te3) / thermo.ctok(te3)
                    tdef2 = (thermo.virtemp(pe4, tp2, tp2) - te2) / \
                        thermo.ctok(te2)
                    lyrf = G * (tdef3 + tdef2) / 2. * (h4 - h3)
                    if lyrf > 0: pcl.b6km += lyrf
        else: pcl.b6km = 0.
        
        h1 = h2

        # LFC Possibility (only if there's a positive energy layer above a negative energy layer.)
        if lyre >= 0. and lyrlast <= 0.:
            tp3 = tp1
            #te3 = te1
            pe2 = pe1
            pe3 = pelast
            # WHY IS THIS LIFTING FROM PE2 TO PE3 USING TP3?
            # OKAY TP3 = TP1 AND PE2 = PE1 AND PE3 = PELAST
            # SO I'M REALLY LIFTING FROM PE1 USING TP1 TO PELAST.
            # WHY AM I CALLING WETLIFT 2 TWICE FOR THE SAME CALCULATION THIS NUMBER OF TIMES?
            # WET LIFT GETS CALLED AT LEAST TWICE WHEN YOU HAVE TO EVALUATE THIS FIRST CONDITIONAL STATEMENT.
            #tp3 = thermo.wetlift(pe2, tp3, pe3, pcl.thetae, method=method)
            #tp3 = thermo.virtemp(pe3, tp3, tp3)
            tp3 = tv_pcl[i-iter_ranges[0]-1]
            if tv_env[i-iter_ranges[0]-1] < tp3:
                # Found an LFC, store height/pres and reset EL/MPL
                pcl.lfcpres = pe3
                pcl.lfchght = interp.to_agl(prof, interp.hght(prof, pe3))
                #print "Found an LFC."
                #print "\tLifting from pe1 to pelast:", pe1, pe3, tp1, tp3
                pcl.elpres = ma.masked
                pcl.elhght = ma.masked
                pcl.mplpres = ma.masked
            else:
                pe_range = np.arange(pe3,pe2-5,-5)
                tp3_range = thermo.wetlift(pe2, tp3, pe_range, pcl.thetae, method=method)
                idx = np.where(interp.vtmp(prof, pe_range) - tp3_range > 0)[0][0]
                pe3 = pe_range[idx]
                #print np.where(interp.vtmp(prof,pe_range) - tp3)
                #while interp.vtmp(prof, pe3) > tp3 and pe3 > 0:
                #    pe3 -= 5
                #    tp3 = thermo.wetlift(pe2, tp3, pe3, pcl.thetae, method=method)
                #    tp3 = thermo.virtemp(pe3, tp3, tp3)[0]
                #    print "\t\tLifting from pe1 to pelast:", pe1, pe3, tp1, tp3
                #print "\t\tLoop broken.", pe3
                if pe3 > 0:
                    # Found a LFC, store height/pres and reset EL/MPL
                    pcl.lfcpres = pe3
                    pcl.lfchght = interp.to_agl(prof, interp.hght(prof, pe3))
                    cinh_old = totn
                    tote = 0.
                    li_max = -9999.
                    if cap_strength < 0.: cap_strength = 0.
                    pcl.cap = cap_strength
                    pcl.cappres = cap_strengthpres

                    pcl.elpres = ma.masked
                    pcl.elhght = ma.masked
                    pcl.mplpres = ma.masked

            # Hack to force LFC to be at least at the LCL
            if pcl.lfcpres >= pcl.lclpres:
                pcl.lfcpres = pcl.lclpres
                pcl.lfchght = pcl.lclhght
                #print "Found an LFC."
                
        # EL Possibility (only if there's a negative energy layer above a positive energy layer.)
        if lyre <= 0. and lyrlast >= 0. and totp > 0:
            #print "Found an EL."
            tp3 = tp1
            #te3 = te1
            pe2 = pe1
            pe3 = pelast
            #tp3 = thermo.wetlift(pe2, tp3, pe3, pcl.thetae, method=method)
            #tp3 = thermo.virtemp(pe3, tp3, tp3)
            tp3 = tv_pcl[i-iter_ranges[0]-1]
            #print "\tLifting from pe1 to pelast:", pe1, pe3, tp1, tp3
            pe_range = np.arange(pe3,pe2-5,-5)
            #print pe_range
            tp3_range = thermo.wetlift(pe2, tp3, pe_range, pcl.thetae, method=method)
            #print tp3_range
            #print interp.vtmp(prof, pe_range) - tp3_range
            idx = np.where(interp.vtmp(prof, pe_range) - tp3_range < 0)[0]
            #print idx
            if len(idx) == 0:
                pe3 = pe3
            else:
                pe3 = pe_range[idx]
            #TODO: Rework this logic to mimic the loop using the Numpy trickery
            #print pe3       
            #while interp.vtmp(prof, pe3) < tp3:
            #    pe3 -= 5
            #    tp3 = thermo.wetlift(pe2, tp3, pe3, pcl.thetae, method=method)
            #    tp3 = thermo.virtemp(pe3, tp3, tp3)
                #print "\tLifting from pe1 to pelast:", pe1, pe3, tp1, tp3
            #print pe3
            #stop
            pcl.elpres = pe3
            pcl.elhght = interp.to_agl(prof, interp.hght(prof, pcl.elpres))
            pcl.mplpres = ma.masked
            pcl.limax = -li_max
            pcl.limaxpres = li_maxpres
        
        # MPL Possibility (only if total energy of the profile is 0 J/kg)
        if tote < 0. and not utils.QC(pcl.mplpres) and utils.QC(pcl.elpres):
            #print "Found an MPL."
            pe3 = pelast
            h3 = interp.hght(prof, pe3)
            te3 = interp.vtmp(prof, pe3)
            tp3 = thermo.wetlift(pe1, tp1, pe3, pcl.thetae, method=method)
            totx = tote - lyre
            pe2 = pelast
            #print "\tLifting from pe1 to pelast:", pe1, pe3, tp1, tp3
            
            while totx > 0:
                pe2 -= 1
                te2 = interp.vtmp(prof, pe2)
                tp2 = thermo.wetlift(pe3, tp3, pe2, pcl.thetae, method=method) # lift from pe3 to pe2
                h2 = interp.hght(prof, pe2)
                tdef3 = (thermo.virtemp(pe3, tp3, tp3) - te3) / \
                    thermo.ctok(te3)
                tdef2 = (thermo.virtemp(pe2, tp2, tp2) - te2) / \
                    thermo.ctok(te2)
                lyrf = G * (tdef3 + tdef2) / 2. * (h2 - h3)
                totx += lyrf
                tp3 = tp2
                te3 = te2
                pe3 = pe2
                #print "\ttotx:",totx, lyrf,tote
                #print "\tLifting from pe3 to pe2:", pe2, pe3, tp2, tp3

            pcl.mplpres = pe2
            pcl.mplhght = interp.to_agl(prof, interp.hght(prof, pe2))
        
        # 500 hPa Lifted Index
        # Should enter this logic one time (as long as the pressure is less than or equal to 500)
        if prof.pres[i] <= 500. and not utils.QC(pcl.li5):
            #print 'Calculating LI5...'
            # Wetlift has no idea that sometimes pe3 == 500 mb.
            # TODO: Need to make it smarter!!!
            if pe1 != 500:
                a = interp.vtmp(prof, 500.)
                b = thermo.wetlift(pe1, tp1, 500.)
                b = thermo.virtemp(500, b, b)
            else:
                a = tv_env[i - iter_ranges[0]]
                b = tv_pcl[i - iter_ranges[0]]
            pcl.li5 = a - b
            # if the profile has a 500 mb level, this means we're calling wetlift twice to get Tw @ 500 mb
        
        # 300 hPa Lifted Index
        # Should enter this logic one time (as long as the pressure is less than or equal to 500)
        if prof.pres[i] <= 300. and not utils.QC(pcl.li3):
            #print 'Calculating LI3...'
            # Wetlift has no idea that sometimes pe3 == 300 mb.
            # TODO: Need to make it smarter!!!
            if pe1 != 300:
                a = interp.vtmp(prof, 300.)
                b = thermo.wetlift(pe1, tp1, 300.)
                b = thermo.virtemp(300, b, b)
            else:
                a = tv_env[i - iter_ranges[0]]
                b = tv_pcl[i - iter_ranges[0]]
            pcl.li3 = a - b
            # if the profile has a 300 mb level, this means we're calling wetlift twice to get Tw @ 300 mb

    
#    pcl.bminus = cinh_old

    if not utils.QC(pcl.bplus): pcl.bplus = totp
    
    # Calculate BRN if available
    params.bulk_rich(prof, pcl)
    
    # Save params
    if np.floor(pcl.bplus) == 0: pcl.bminus = 0.
    pcl.ptrace = ma.concatenate((ptrace, ptraces))
    pcl.ttrace = ma.concatenate((ttrace, tv_pcl))

    # Find minimum buoyancy from Trier et al. 2014, Part 1
    idx = np.ma.where(pcl.ptrace >= 500.)[0]
    if len(idx) != 0:
        b = pcl.ttrace[idx] - interp.vtmp(prof, pcl.ptrace[idx])
        idx2 = np.ma.argmin(b)
        pcl.bmin = b[idx2]
        pcl.bminpres = pcl.ptrace[idx][idx2]
    
    return pcl

#spc_file = open('14061619.OAX', 'r').read()
#pres, hght, tmpc, dwpc, wdir, wspd = parseSPC(spc_file)

from netCDF4 import Dataset
d = Dataset('sgpsondewnpnC1.b1.20130520.113100.cdf')
d = Dataset('sgpsondewnpnC1.b1.20130520.172600.cdf')
tmpc = d.variables['tdry'][:]
pres = d.variables['pres'][:]
hght = d.variables['alt'][:]
dwpc = d.variables['dp'][:]
wspd = d.variables['wspd'][:]
wdir = d.variables['deg'][:]

prof = profile.create_profile(profile='default', pres=pres, hght=hght, tmpc=tmpc, \
                                    dwpc=dwpc, wspd=wspd, wdir=wdir, missing=-9999, strictQC=False)

from datetime import datetime
dt = datetime.now()
pcl1 = params.parcelx(prof, flag=3)
tfl1 = datetime.now() - dt
print pcl1.bplus, pcl1.bminus, pcl1.lclhght, pcl1.lfchght, pcl1.elhght
print "Time for Lift:",tfl1 
dt = datetime.now()
pcl2 = parcelx(prof,flag=3)
tfl2 = datetime.now() - dt
print pcl2.bplus, pcl2.bminus, pcl2.lclhght, pcl2.lfchght, pcl2.elhght
print "Time for Lift:", tfl2
#for i in range(len(pcl2.ttrace)):
#    print pcl1.ttrace[i], pcl2.ttrace[i], pcl1.ptrace[i], pcl2.ptrace[i]
from pylab import *
from matplotlib.ticker import (MultipleLocator, NullFormatter, ScalarFormatter)

fig = figure(figsize=(6.5875, 6.2125))
ax = fig.add_subplot(111, projection='skewx')

grid(True)

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
ax.semilogy(prof.tmpc, prof.pres, color='r')
ax.semilogy(prof.vtmp, prof.pres, color='r', linestyle='--')
ax.semilogy(prof.wetbulb, prof.pres, color='c', linestyle='-')
ax.semilogy(prof.dwpc, prof.pres, color='g')
ax.semilogy(pcl1.ttrace, pcl1.ptrace, color='k', linestyle='--', label='WOBUS ' + str(tfl1) + ' s')
ax.semilogy(pcl2.ttrace, pcl2.ptrace, color='m', linestyle='--', label='RDJ ' + str(tfl2) + ' s')
legend()
# An example of a slanted line at constant X
l = ax.axvline(0, color='C0')

# Disables the log-formatting that comes with semilogy
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks(np.linspace(100, 1000, 10))
ax.set_ylim(1050, 100)

ax.xaxis.set_major_locator(MultipleLocator(10))
ax.set_xlim(-50, 50)
show()
