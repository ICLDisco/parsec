import pandas as pd
import sys
import os

# oringinally developed by Daniel Mishler in the fall of 2023.

# take a trace from v1, and make an "events" df matching v2's "events" fields
def tv2_events_from_tv1_trace(v1trace):
    v1events = v1trace['events']
    tv2_aei = {} # all event info
    always_present = []
    
    for col in v1events.columns:
        if any(v1events[col].isnull()) == False:
            always_present.append(col)
    
    # could skip above loop for time and opt for defaults in a pinch
    """
    always_present_default = ['node_id', 'stream_id', 'taskpool_id', 'type', 'begin', 'end', 'flags', 'id']
    always_present = always_present_default
    """
    
    tv2_aei['events'] = v1events[always_present]
    
    
    other_columns = []
    for item in v1events.columns:
        if item not in always_present:
            other_columns.append(item)
    for i in range(len(v1trace['event_types'])):
        sliced_df = v1events[v1events.type == i]

        relevant_columns = other_columns.copy()        
        for col in other_columns:
            if all(sliced_df[col].isnull()) == True: # Everything is null
                relevant_columns.remove(col)
        
        tv2_aei[f"event_infos_{i}"] = sliced_df[relevant_columns]
    
    return tv2_aei

# take a trace from v2, and make an "events" df matching v1's "events"
def tv1_events_from_tv2_trace(v2trace):
    events = v2trace['events']
    new_cols = set()
    for i in range(len(v2trace['event_types'])):
        for col in v2trace[f'event_infos_{i}']:
            new_cols.add(col)
    for col in new_cols:
        events[col] = None
    for i in range(len(v2trace['event_types'])):
        events.update(v2trace[f'event_infos_{i}'])
    return events


# this function may not be needed on a large scale, but it was a utility
# used in development
def compare_entries(a, b):
    if type(a) not in [pd.core.frame.DataFrame, pd.core.series.Series]:
        print("CE ERROR: unknown type for a")
        return False
    if type(b) not in [pd.core.frame.DataFrame, pd.core.series.Series]:
        print("CE ERROR: unknown type for b")
        return False
    if(len(a) != len(b)):
        print("CE ERROR: lengths do not match")
        return False
    if type(a) is pd.core.frame.DataFrame and type(b) is pd.core.frame.DataFrame:
        # try a silly converstion?
        a = a.astype(str)
        b = b.astype(str) # this is digusting that this works. Consider a better option?
        # realign axes 
        try:
            a = a[b.columns]
        except KeyError:
            b = b[a.columns]
        comparison_obj = a.compare(b)
    else:
        comparison_obj = (a == b)
    if(len(comparison_obj) == 0):
        return True
    if(type(comparison_obj) is pd.core.series.Series):
        if comparison_obj.any() == False:
            return False
        else:
            return True
    if(type(comparison_obj) is pd.core.frame.DataFrame):
        for key in comparison_obj:
            if comparison_obj[key].any() == False:
                return False
        # else
        return True

def sanity_check_traces(ta, tb):
    ta_keys = ta.keys()
    for key in ta_keys:
        if key in tb.keys():
            match = compare_entries(ta[key], tb[key])
            if match:
                print(f"key {key} matches")
            else:
                print(f"key {key} does NOT match")

def sanity_check_files(f1name, f2name):
    f1trace = pd.HDFStore(f1name)
    f2trace = pd.HDFStore(f2name)
    
    sanity_check_traces(f1trace,f2trace)
    
    f1trace.close()
    f2trace.close()
                
        

def tv2_to_tv1(infile, outfile):
    v2trace = pd.HDFStore(infile)
    v1trace = pd.HDFStore(outfile)
    
    for key in v2trace:
        if "/events" == key or "event_infos" in key:
            continue
        if "/information" == key:
            v2trace[key].to_hdf(outfile, key=key, mode='a')
            continue
        if "/nodes" == key:
            continue
        # else
        v1trace.append(key, v2trace[key])
    
    
    tv1events = tv1_events_from_tv2_trace(v2trace)
    
    
    for col in tv1events:
        tv1events[col] = tv1events[col].astype(str)
        
    v1trace.append("events", tv1events)
    
    
    v2trace.close()
    v1trace.close()
    
def tv1_to_tv2(infile, outfile):
    v1trace = pd.HDFStore(infile)
    
    for key in v1trace:
        if "/events" == key or "event_infos" in key:
            continue
        # else
        v1trace[key].to_hdf(outfile, key=key, mode='a')
    
    
    # now handle events
    tv2_aei = tv2_events_from_tv1_trace(v1trace)
    for key in tv2_aei:
        tv2_aei[key].to_hdf(outfile, key=key, mode='a')
    
    v1trace.close()
    
def help_message():
    print(f"usage: <python> {sys.argv[0]} <version_option> <input_hdf5> <output_hdf5>")
    print(f"    version_options:")
    print(f"        2t1 // converts from version 2 to version 1")
    print(f"        1t2 // converts from version 1 to version 2")
    

if __name__ == "__main__":
    if(len(sys.argv) < 4):
        help_message()
        exit()
    
    
    version_option = sys.argv[1]
    infile = sys.argv[2]
    outfile = sys.argv[3]
    
    if(os.path.isfile(outfile)):
        print(f"`{outfile}` previously existed and will be overwritten.")
        # overwrite the output file
        os.remove(outfile)
        # alternatively, add this in for a comfirmation message
        """
        print(f"`{outfile}` is a file that already exists. Do you wish to overwrite it? (yes/no)")
        uinput = input()
        if uinput == "y" or uinput == "yes":
            # overwrite the output file
            os.remove(outfile)
        else:
            print("halting execution")
            exit()
        """
    
    if(version_option == "2t1"):
        tv2_to_tv1(infile, outfile)
    elif(version_option == "1t2"):
        tv1_to_tv2(infile, outfile)
    else:
        print("error: could not understand <version_option>")
        help_message()
        exit()
        
    """
    print("SANITY CHECK")
    if (version_option == "2t1"):
        sanity_check_files(infile, outfile)
    elif (version_option == "1t2"):
        sanity_check_files(infile, outfile)
    """