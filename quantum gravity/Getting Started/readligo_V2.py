"""
readligo.py
Version 0.2
April 21, 2016
Jonah Kanner, Roy Williams, and Alan Weinstein

Updates in this version:
 * Should now work with both Python 2 and Python 3

This module provides tools for reading LIGO data
files.  Data along with supporting documentation
can be downloaded from the losc web site:
https://losc.ligo.org

Some possible use cases are shown below.

Example #0:
To load all data from a single file:
strain, time, dq = rl.loaddata('ligo_data/H-H1_LOSC_4_V1-842653696-4096.hdf5', 'H1')

Example #1: 
segList = getsegs(842657792, 842658792, 'H1')
for (start, stop) in segList:
  strain, meta, dq = getstrain(start, stop, 'H1')
  # -- Analysis code here
  ...

This default configuration assumes that the needed LIGO data 
files are available in the current working directory or a 
subdirectory.  LIGO data between the input GPS times is loaded
into STRAIN.  META is a dictionary of gps start, gps stop, and the 
sample time.  DQ is a dictionary of data quality flags.

Example #2
segList = SegmentList('H1_segs.txt')

In Example 2, 'H1_segs.txt' is a segment list downloaded from the
LOSC web site using the Timeline application.  This may be used in the same
manner as segList in example 1.

Example #3
filelist = FileList(directory='/home/ligodata')
segList = getsegs(842657792, 842658792, 'H1', filelist=filelist)
for start, stop in segList:
  strain, meta, dq = getstrain(start, stop, 'H1', filelist=filelist)
  # -- Analysis code here

In this example, the first command searches the indicated directory and 
sub-directories for LIGO data files.  This list of data files is then 
used to construct a segment list and load the requested data.  

-- SEGMENT LISTS --

Segment lists may be downloaded from the LOSC web site
using the Timeline Query Form or constructed directly
from the data files.  

Read in a segment list downloaded from the Timeline 
application on the LOSC web site with SegmentList:
>> seglist = SegmentList('H1_segs.txt')
OR
Construct a segment list directly from the LIGO
data files with getsegs():
>> seglist = getsegs(842657792, 842658792, 'H1', flag='DATA', filelist=None)

"""

import numpy as np
import os
import fnmatch

def read_frame(filename, ifo, readstrain=True):
    """
    Helper function to read frame files
    """
    try:
        import Fr
    except:
        from pylal import Fr

    if ifo is None:
        raise TypeError("""To read GWF data, ifo must be 'H1', 'H2', or 'L1'.
        def loaddata(filename, ifo=None):""")

    #-- Read strain channel
    strain_name = ifo + ':LOSC-STRAIN'
    if readstrain:
        sd = Fr.frgetvect(filename, strain_name)    
        strain = sd[0]
        gpsStart = sd[1] 
        ts = sd[3][0]
    else:
        ts = 1
        strain = 0
    
    #-- Read DQ channel
    dq_name = ifo + ':LOSC-DQMASK'
    qd = Fr.frgetvect(filename, dq_name)
    gpsStart = qd[1]
    qmask = np.array(qd[0])
    dq_ts = qd[3][0]
    shortnameList_wbit = qd[5].split()
    shortnameList = [name.split(':')[1] for name in shortnameList_wbit]

    #-- Read Injection channel
    inj_name = ifo + ':LOSC-INJMASK'
    injdata = Fr.frgetvect(filename, inj_name)
    injmask = injdata[0]
    injnamelist_bit = injdata[5].split()
    injnamelist     = [name.split(':')[1] for name in injnamelist_bit]

    return strain, gpsStart, ts, qmask, shortnameList, injmask, injnamelist
    
def read_hdf5(filename, readstrain=True):
    """
    Helper function to read HDF5 files
    """
    import h5py
    dataFile = h5py.File(filename, 'r')

    #-- Read the strain
    if readstrain:
        strain = dataFile['strain']['Strain'][...]
    else:
        strain = 0

    ts = dataFile['strain']['Strain'].attrs['Xspacing']
    
    #-- Read the DQ information
    dqInfo = dataFile['quality']['simple']
    qmask = dqInfo['DQmask'][...]
    shortnameArray = dqInfo['DQShortnames'].value
    shortnameList  = list(shortnameArray)
    
    # -- Read the INJ information
    injInfo = dataFile['quality/injections']
    injmask = injInfo['Injmask'][...]
    injnameArray = injInfo['InjShortnames'].value
    injnameList  = list(injnameArray)
    
    #-- Read the meta data
    meta = dataFile['meta']
    gpsStart = meta['GPSstart'].value    
    
    dataFile.close()
    return strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList

def loaddata(filename, ifo=None, tvec=True, readstrain=True):
    """
    The input filename should be a LOSC .hdf5 file or a LOSC .gwf
    file.  The file type will be determined from the extenstion.  
    The detector should be H1, H2, or L1.

    The return value is: 
    STRAIN, TIME, CHANNEL_DICT

    STRAIN is a vector of strain values
    TIME is a vector of time values to match the STRAIN vector
         unless the flag tvec=False.  In that case, TIME is a
         dictionary of meta values.
    CHANNEL_DICT is a dictionary of data quality channels    
    """

    # -- Check for zero length file
    if os.stat(filename).st_size == 0:
        return None, None, None

    file_ext = os.path.splitext(filename)[1]    
    if (file_ext.upper() == '.GWF'):
        strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList = read_frame(filename, ifo, readstrain)
    else:
        strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList = read_hdf5(filename, readstrain)
        
    #-- Create the time vector
    gpsEnd = gpsStart + len(qmask)
    if tvec:
        time = np.arange(gpsStart, gpsEnd, ts)
    else:
        meta = {}
        meta['start'] = gpsStart
        meta['stop']  = gpsEnd
        meta['dt']    = ts

    #-- Create 1 Hz DQ channel for each DQ and INJ channel
    channel_dict = {}  #-- 1 Hz, mask
    slice_dict   = {}  #-- sampling freq. of stain, a list of slices
    final_one_hz = np.zeros(qmask.shape, dtype='int32')
    for flag in shortnameList:
        bit = shortnameList.index(flag)
        # Special check for python 3
        if isinstance(flag, bytes): flag = flag.decode("utf-8") 
        
        channel_dict[flag] = (qmask >> bit) & 1

    for flag in injnameList:
        bit = injnameList.index(flag)
        # Special check for python 3
        if isinstance(flag, bytes): flag = flag.decode("utf-8") 
        
        channel_dict[flag] = (injmask >> bit) & 1
       
    #-- Calculate the DEFAULT channel
    try:
        channel_dict['DEFAULT'] = ( channel_dict['DATA'] )
    except:
        print("Warning: Failed to calculate DEFAULT data quality channel")

    if tvec:
        return strain, time, channel_dict
    else:
        return strain, meta, channel_dict


def dq2segs(channel, gps_start):
    """
    This function takes a DQ CHANNEL (as returned by loaddata or getstrain) and 
    the GPS_START time of the channel and returns a segment
    list.  The DQ Channel is assumed to be a 1 Hz channel.

    Returns of a list of segment GPS start and stop times.
    """
    #-- Check if the user input a dictionary
    if type(channel) == dict:
        try:
            channel = channel['DEFAULT']
        except:
            print("ERROR: Could not find DEFAULT channel in dictionary")
            raise

    #-- Create the segment list
    segments = dq_channel_to_seglist(channel, fs=1)
    t0 = gps_start
    segList = [(int(seg.start+t0), int(seg.stop+t0)) for seg in segments]
    return SegmentList(segList)
    
def dq_channel_to_seglist(channel, fs=4096):
    """
    WARNING: 
    This function is designed to work the output of the low level function
    LOADDATA, not the output from the main data loading function GETSTRAIN.

    Takes a data quality 1 Hz channel, as returned by
    loaddata, and returns a segment list.  The segment
    list is really a list of slices for the strain 
    associated strain vector.  

    If CHANNEL is a dictionary instead of a single channel,
    an attempt is made to return a segment list for the DEFAULT
    channel.  

    Returns a list of slices which can be used directly with the 
    strain and time outputs of LOADDATA.
    """
    #-- Check if the user input a dictionary
    if type(channel) == dict:
        try:
            channel = channel['DEFAULT']
        except:
            print("ERROR: Could not find DEFAULT channel in dictionary")
            raise

    # -- Create the segment list
    condition = channel > 0
    boundaries = np.where(np.diff(condition) == True)[0]
    # -- Need to +1 due to how np.diff works 
    boundaries = boundaries + 1
    # if the array "begins" True, we need to complete the first segment
    if condition[0]:
        boundaries = np.append(0,boundaries)
    # if the array "ends" True, we need to complete the last segment
    if condition[-1]:
        boundaries = np.append(boundaries,len(condition))

    # -- group the segment boundaries two by two
    segments = boundaries.reshape((len(boundaries)/2,2))
    # -- Account for sampling frequency and return a slice
    segment_list = [slice(start*fs, stop*fs) for (start,stop) in segments]
    
    return segment_list

class FileList():
    """
    Class for lists of LIGO data files.
    
    When a FileList instance is created, DIRECTORY will 
    be searched for LIGO data files.  Sub-directories
    will be searched as well.  By default, the current
    working directory is searched.  
    """
    def __init__(self, directory=None, cache=None):

        # -- Set default directory
        if directory is None:
            if os.path.isdir('/archive/losc/strain-gwf'):
                directory='/archive/losc/strain-gwf'
            else:
                directory='.'

        print("Using data directory {0} ...".format(directory))
        self.directory = directory
        self.cache = cache
        if cache is None:
            self.list = self.searchdir(directory)
        else:
            self.readcache()

    def searchdir(self, directory='.'):
        frameList = []
        hdfList   = []
        for root, dirnames, filenames in os.walk(directory):
            for filename in fnmatch.filter(filenames, '*.gwf'):
                frameList.append(os.path.join(root, filename))
            for filename in fnmatch.filter(filenames, '*.hdf5'):
                hdfList.append(os.path.join(root, filename))
        return frameList + hdfList

    def writecache(self, cacheName):
        outfile = open(cacheName, 'w')
        for file in self.list:
            outfile.write(file + '\n')
        outfile.close()

    def readcache(self):
        infile = open(self.cache, 'r')
        self.list = infile.read().split()
        infile.close()
    
    def findfile(self, gps, ifo):
        start_gps = gps - (gps % 4096)
        filenamelist = fnmatch.filter(self.list, '*' + '-' + ifo + '*' + '-' + str(start_gps) + '-' + '*')
        if len(filenamelist) == 0:
            print("WARNING!  No file found for GPS {0} and IFO {1}".format(gps, ifo))
            return None
        else:
            return filenamelist[0]
            
def getstrain(start, stop, ifo, filelist=None):
    """
    START should be the starting gps time of the data to be loaded.
    STOP  should be the end gps time of the data to be loaded.
    IFO should be 'H1', 'H2', or 'L1'.
    FILELIST is an optional argument that is a FileList() instance.

    The return value is (strain, meta, dq)
    
    STRAIN: The data as a strain time series
    META: A dictionary of meta data, especially the start time, stop time, 
          and sample time
    DQ: A dictionary of the data quality flags
    """

    if filelist is None:
        filelist = FileList()

    # -- Check if this is a science segment
    segList = getsegs(start, stop, ifo, flag='DATA', filelist=filelist)
    sl = segList.seglist
    if (sl[0][0] == start) and (sl[0][1] == stop):
        pass
    else:
        raise TypeError("""Error in getstrain.
        Requested times include times where the data file was not found
        or instrument not in SCIENCE mode.
        Use readligo.getsegs() to construct a segment list.
        The science mode segment list for the requested time range is: 
        {0}""".format(segList))

    # -- Construct list of expected file start times
    first = start - (start % 4096)
    gpsList = np.arange(first, stop, 4096)

    m_strain = np.array([])
    m_dq     = None
    # -- Loop over needed files
    for time in gpsList:
        filename = filelist.findfile(time, ifo)
        print("Loading {0}".format(filename))

        #-- Read in data
        strain, meta, dq = loaddata(filename, ifo, tvec=False)
        if len(m_strain) == 0:
            m_start = meta['start']
            dt = meta['dt']
        m_stop = meta['stop']
        m_strain = np.append(m_strain, strain)
        if m_dq is None:
            m_dq = dq
        else:
            for key in dq.keys():
                m_dq[key] = np.append(m_dq[key], dq[key])

    # -- Trim the data
    lndx  = np.abs(start - m_start)*(1.0/dt)
    rndx = np.abs(stop - m_start)*(1.0/dt)
        
    m_strain = m_strain[lndx:rndx]
    for key in m_dq.keys():
        m_dq[key] = m_dq[key][lndx*dt:rndx*dt]

    meta['start'] = start
    meta['stop']  = stop
    meta['dt']    = dt

    return m_strain, meta, m_dq

class SegmentList():
    def __init__(self, filename, numcolumns=3):

        if type(filename) is str:
            if numcolumns == 4:
                number, start, stop, duration = np.loadtxt(filename, dtype='int',unpack=True)
            elif numcolumns == 2:
                start, stop = np.loadtxt(filename, dtype='int',unpack=True)
            elif numcolumns == 3:
                start, stop, duration = np.loadtxt(filename, dtype='int',unpack=True)
            self.seglist = zip(start, stop)
        elif type(filename) is list:
            self.seglist = filename
        else:
            raise TypeError("SegmentList() expects the name of a segmentlist file from the LOSC website Timeline")

    def __repr__(self):
        return 'SegmentList( {0} )'.format(self.seglist)
    def __iter__(self):
        return iter(self.seglist)
    def __getitem__(self, key):
        return self.seglist[key]
       
def getsegs(start, stop, ifo, flag='DATA', filelist=None):
    """
    Method for constructing a segment list from 
    LOSC data files.  By default, the method uses
    files in the current working directory to 
    construct a segment list.  

    If a FileList is passed in the flag FILELIST,
    then those files will be searched for segments
    passing the DQ flag passed as the FLAG argument.
    """

    if filelist is None:
        filelist = FileList()

    # -- Construct list of expected file start times
    first = start - (start % 4096)
    gpsList = np.arange(first, stop, 4096)
    m_dq     = None
    
    # -- Initialize segment list
    segList = []

    # -- Loop over needed files
    for time in gpsList:
        filename = filelist.findfile(time, ifo)

        #-- Read in data
        if filename is None:
            print("WARNING! No file found with GPS start time {0}".format(time))
            print("Segment list may contain errors due to missing files.")
            continue
        else:
            try:
                strain, meta, dq = loaddata(filename, ifo, tvec=False, readstrain=False)     
            except:
                print("WARNING! Failed to load file {0}".format(filename))
                print("Segment list may contain errors due to corrupt files.")
                continue

        if dq is None:
            print("Warning! Found zero length file {0}".format(filename))
            print("Segment list may contain errors.")
            continue

        #-- Add segments to list on a file-by-file basis
        chan = dq[flag]
        indxlist = dq_channel_to_seglist(chan, fs=1.0)
        i_start = meta['start']
        i_seglist = [(indx.start+i_start, indx.stop+i_start) for indx in indxlist]
        i_seglist = [(int(begin), int(end)) for begin, end in i_seglist] 
        segList = segList + i_seglist
      
    # -- Sort segments
    segList.sort()
    
    # -- Merge overlapping segments
    for i in range(0, len(segList)-1):
        seg1 = segList[i]
        seg2 = segList[i+1]
    
        if seg1[1] == seg2[0]:
            segList[i]   = None
            segList[i+1] = (seg1[0], seg2[1])            
    # -- Remove placeholder segments
    segList = [seg for seg in segList if seg is not None]

    # -- Trim segment list to fit within requested start/stop times
    for seg in segList:
        idx = segList.index(seg)
        if (seg[1] < start):
            segList[idx] = None
        elif (seg[0] > stop):
            segList[idx] = None
        elif (seg[0] < start) and (seg[1] > stop):
            segList[idx] = (start, stop)
        elif (seg[0] < start):
            segList[idx] = (start, seg[1])
        elif (seg[1] > stop):
            segList[idx] = (seg[0], stop)
    # -- Remove placeholder segments
    segList = [seg for seg in segList if seg is not None]

    return SegmentList(segList)


