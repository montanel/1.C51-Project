import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#create useful variables
idx = pd.IndexSlice
rng = np.random.default_rng()
plt_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams['svg.fonttype'] = 'none' #prevent matplotlib to save every letter separately in svg
plt.rcParams['figure.facecolor'] = 'white' #make figure background white not transparent (for the sole purpose of sharing images in slack dark mode)
porm = u"\u00B1"

states_dict = {'01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE', 
               '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA', 
               '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', 
               '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM', 
               '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '72': 'PR', 
               '44': 'RI', '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', 
               '78': 'VI', '53': 'WA', '54': 'WV', '55': 'WI', '56': 'WY'}

class IterTimer():
    '''Timer for iterations. Usage:
    
    n_iters = 50
    timer = IterTimer(n_iters=n_iters, ...)
    
    for args in arguments:
        timer.start_iter()
        your_function(args)
        t_end, mean_iter = timer.end_iter()
    '''
    
    def __init__(self, n_iters, default_print=True, window=0, log_times=True, log_folder=None):
        '''
        args:
            n_iters: (int) Total number of iterations.
            default_print: (bool) Whether to let IterTimer print at the end of the iteration. 
                           WARNING: uses in-place priting that might interfere with other printing (default: True).
            window: (int) Number of iterations to consider when making the average, -10 is the last 10 iterations, 0 is all (default: 0).
            log_times: (bool) Whether to save the iteration times in an .npy file, useful to make the ETA prediciton code better (default: True).
            log_folder: (str) Folder location of where to save iteration times, using an absolute path is recommended. 
                        None will search for some default paths (default: None).
        '''
        
        self.iter_times = []
        self.n_iters = n_iters
        self.default_print = default_print
        self.window = window
        self.current_iter = 0
        self.log_times = log_times
        
        #if log_folder is None, we'll see if some default values exist
        if self.log_times and log_folder is None:
            if os.path.exists("F:/Code/misc/IterTimer/timer_data"): #propus
                log_folder = "F:/Code/misc/IterTimer/timer_data"
            # elif os.path.exists("C:/Code/misc/IterTimer/timer_data"): #mebsuta
            #     log_folder = "C:/Code/misc/IterTimer/timer_data"
            elif os.path.exists("timer_data"): #generic
                log_folder = os.path.abspath("timer_data")
            elif os.path.exists("../timer_data"): #generic
                log_folder = os.path.abspath("../timer_data")
            elif os.path.exists("../../timer_data"): #generic
                log_folder = os.path.abspath("../../timer_data")
            else: #if nothing then we don't log
                self.log_times = False

        #if we have a log_folder of we found a default location, let's set up
        if self.log_times:
            if not os.path.exists(log_folder): #in the case the user defined log_folder doesn't exist
                raise Exception(f"Logging path {log_folder} was not found.")
            import uuid
            self.save_file = f"{log_folder}/{uuid.uuid4()}.{self.n_iters}.npy"
    
    def start_iter(self):
        '''Start measuring time.'''
        self.start = time.perf_counter()
    
    def end_iter(self):
        '''Finish measuring time and calculate relevant metrics. Will print if defaul_print=True.
        
        args:
            default_print: (bool) Whether to let IterTimer print at the end of the iteration, overwrites Class default_print (default: False).
            
        returns:
            t_end: (float) Absolute ETA in seconds.
            mean_iter: (float) Mean iteration time.
        '''
        
        #keep track of time and calculate ETA based on mean iteration time
        self.iter_times.append(time.perf_counter()-self.start)
        self.mean_iter = np.array(self.iter_times[self.window:]).mean() #only take the last few iterations to calculate mean to account for changing conditions
        self.t_end = time.time() + self.mean_iter*(self.n_iters-self.current_iter-1)
        
        #if default_print call the function to print
        if self.default_print: self.print_default()
        
        #incremenent the number of iterations
        self.current_iter += 1
        
        #saving the iter times just to get some data about what I tend to measure to find best method
        if self.log_times: np.save(self.save_file, self.iter_times)
        
        return self.t_end, self.mean_iter
    
    def print_default(self):
        '''Default printing routine.'''
        if self.current_iter+1 == self.n_iters:
            print(f"Iteration {self.current_iter+1}/{self.n_iters}, ETA: {self.get_time(time.localtime(self.t_end))} (average time per iteration: {self.mean_iter:.3f}s)")
        else:
            print(f"Iteration {self.current_iter+1}/{self.n_iters}, ETA: {self.get_time(time.localtime(self.t_end))} (average time per iteration: {self.mean_iter:.3f}s)", end="\r")
            
    def get_time(self, time):
        '''Converts a time.struct_time into HH:mm:ss.'''
        return f"{time.tm_hour:02}:{time.tm_min:02}:{time.tm_sec:02}"
    
def res_str(vals, std_err=95, range_factor=1, units=None, range_units=None):
    '''Prints the way I typically print results (for model fitting or timing of code usually).
    
    args:
        vals: (list) List of values to plot mean and range for.
        std_err: (bool) If False, prints the standard deviation, if a float, prints the nth confidence bound (default: 95).
        units: (str) The string for the units to print (default: None).
    '''
    
    if range_factor!=1 and units is not None and range_units is None:
        raise Exception("range_units must have a value if range_factor is 1 and you provided units.")
        
    if range_factor==1:
        if units is None and range_units is not None: units = range_units
        else: range_units = units
        
    vals_np = np.array(vals)
    if std_err is not None:
        val_range = stats.norm.ppf(0.5+std_err/200)*vals_np.std()/np.sqrt(len(vals_np))
    else:
        val_range = vals_np.std()
        
    if units is None:
        return f"{vals_np.mean():.4f} {porm} {range_factor*val_range:.4f}"
    elif str==type(units):
        return f"{vals_np.mean():.4f} {units} {porm} {range_factor*val_range:.4f} {range_units}"