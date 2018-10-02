# SwissFEL_ana
Data analysis: HowTo

The analysis is performed by python scripts. It revolves around the dask library to do lazy computation and parallelization. http://dask.pydata.org/en/latest/

Login to Ra and get a node:
https://www.psi.ch/photon-science-data-services/offline-computing-facility-for-sls-and-swissfel-data-analysis

The Bernina python environment should be used.
To setup environment for data analysis type:
source /sf/bernina/bin/anaconda_env

Scripts files:
config.py,
Process_data.py,
Scan_diagnostic.py,
sf_ana_tools.py,
additional customs analysis function depending on the experiment.

In sf_ana_tools.py, all the main functions are provided.
In particular:

class experiment:
This class holds the main paths and the prefix to save the data

class sf_scan:
This class holds the information about the scans to be analyzed. Its main purpose is to find the right file to be read. It is initiated by taking the experiment class as an input. File to be analyzed can be added to the list by using the add_file method. This method can take either a full name or a scan number as input.
Remark: it was initially thought to be used only scans that are to be averaged together (it checks for the scan motor and throw an error if it does not match). This could be changed maybe.

class detector:
This class hold the info about how to analyze the data from a specific detector:
Its name and its alias in the HDF5 data file
The path to its gain and pedestal files
Pixel intensity filters
The function to be used for its analysis (any functions that take a stack of images as a first input and keywords arguments after)
The list of keywords arguments and  their value
    
class JFdata:
This class contains the Jungfrau (JF) data sets and is at the core of the analysis. Its input consist of a set of raw JF images (JFdata.raw) and its corresponding detector class.
Upon initiation, it will automatically apply in the following order:
Pedestal and gain correction (JFdata.corrected)
Common mode correction (JFdata.CMcorr)
Pixel intensity filters. (JFdata.filtered)
Essentially it also contains the method run_ana, which will apply the custom analysis function to the stack of corrected images.
All of this is applied in a lazy way, meaning that none of the computation are actually performed.
Before the common mode correction, the images are in a dask.array.Array format, which is convenient to perform array operation. However, the common mode correction script does not work with dask array, and from then on it is converted into a series of delayed objects. See dask documentation for more information.

Other functions:
These are some pretty self-explanatory functions that are mainly used by the different classes above. Some analysis function are there too, such as roi_bkgRoi, which will take a region of interest and a background from an image.

These functions and class are then used into the two front-end scripts: Process_data.py and Scan_diagnostic.py.
Those two files are very similar: while one run the full analysis and saves each processed step in a separate hdf5 file, the second one analysis only a given step and provides diagnostics. Typically the second script is run first to check that the parameters (filters, roi, …) are correct. The analysis is then performed by running Process_data.py.
Without going into too much details, these scripts read the data, check the events ids and run the analysis functions. It will loop through all defined detectors and analyze each of them with its predefined analysis function. Finally each step is saved in a hdf5 file.


About the custom analysis function:
The user-supplied analysis function must take a stack of images as their first input. The other argument are given via keywords arguments (**keywords)
It is also important to add the decorator @delayed to the definition of the function, as this is essential for the parallelization. Make sure you have imported the function too (from dask import delayed).
At the moment, the output is assumed to be a single numpy array (with arbitrary numbers of dimensions). It would maybe be nice to output a dictionary if more than one output is desired. However, the function is applied for each delayed object separately, containing 5-10 images typically. A function to put together all the output dictionaries would then be needed.
