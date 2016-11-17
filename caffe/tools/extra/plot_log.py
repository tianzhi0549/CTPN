#!/usr/bin/env python
import sys, re, os
import matplotlib.pyplot as plt
import numpy as np

def parse_keys_indices(keys):
    indices=[]
    real_keys=[]
    pattern=re.compile('(.+)\(\)')

def load_data_from_log_file(log_file, keys, line_filter):
    results={}
    
    regex_iteration=re.compile('Iteration (\d+)')
    
    regexes=[]
    for key in keys:
        regexes.append(re.compile(r' %s( =|:) ([^ ]+)[$\n ]'%key))

    fp=open(log_file)
    iterations=[]
    iter=None
    for line in fp:
        if not line_filter(line):
            continue
        m=regex_iteration.search(line)
        if m:
            iter=int(m.group(1))
            if iter not in results:
                results[iter]={}

        if iter!=None:
            for index, regex in enumerate(regexes):
                m=regex.search(line)
                if m:
                    num=0
                    key=keys[index]
                    while key in results[iter]:
                        num+=1
                        key="%s(%d)"%(keys[index], num)
                    results[iter][key]=float(m.group(2))
    fp.close()
    
    return format(results)

def format(iteration):
    results={}
    for iter, values in iteration.items():
        for key, v in values.items():
            if key not in results:
                results[key]=[]
            results[key].append((iter, v))
    return results

def line_filter_test(line):
    if line.find(" Test ")>=0 or line.find(" Testing ")>=0:
        return True
    else:
        return False

def line_filter_train(line):
    if line_filter_test(line):
        return False
    if line.find("MultiStep")>=0:
        return False
    return True

def plot_log(log_file, keys):
    filename=log_file# os.path.splitext(os.path.basename(log_file))[0]
    train_data=load_data_from_log_file(log_file, keys, line_filter_train)
    test_data=load_data_from_log_file(log_file, keys, line_filter_test)
    
    legends=[]
    for k, data in train_data.items():
        data=zip(*sorted(data, key=lambda x: x[0]))
        plt.plot(data[0], data[1], 'x', linewidth=0.75)
        legends.append("Train %s/%s"%(k, filename))
    for k, data in test_data.items():
        data=zip(*sorted(data, key=lambda x: x[0]))
        plt.plot(data[0], data[1], '-x', linewidth=0.75)
        legends.append("Test %s/%s"%(k, filename))
    return legends

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("log_files", nargs='+')
    parser.add_argument("-k", nargs='+', default=['loss'])
    args=parser.parse_args()
    log_files=args.log_files
    keys=args.k
    legends=[]
    for index in range(0, len(log_files)):
        legends.extend(plot_log(log_files[index], keys))
    plt.legend(legends, 'best')
    plt.show()

if __name__=="__main__":
    main()
