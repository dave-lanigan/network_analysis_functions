import zen
import numpy
from numpy import zeros
import netcontrolz
import networkx
import multiprocessing
import os
import time
#from perc_betweenness import edge_percolation_betweenness_


def highest_betweenness(G):

    """

    Returns the edge index that has the highest betweeness centrality.

    Currently we rely on the netowrkx to calculate this statistic.

    """

    Gnx = zen.nx.to_networkx(G)

    edge_btw = networkx.edge_betweenness_centrality(Gnx,normalized = False)

    high_btw_endpts = sorted(edge_btw.items(), key=lambda x: x[1],reverse = True)[0][0]

    eidx = G.edge_idx(high_btw_endpts[0],high_btw_endpts[1])

    return eidx


def execute(G,param,attackID,showData=True,showTime=True):
    
    list_net_attacks = [netcontrolz.EDGE_ATTACK_INOUT_DEG,netcontrolz.EDGE_ATTACK_OUTIN_DEG,netcontrolz.ATTACK_RAND,netcontrolz.EDGE_ATTACK_TOTAL_DEG]
    
    list_controls = [netcontrolz.CONTROL_ROBUSTNESS,netcontrolz.REACHABILITY_ROBUSTNESS_FIXED,netcontrolz.REACHABILITY_ROBUSTNESS_FREE]
    profiles = []
    
    files_o = param[5]
    start_time=param[4]
    num_files_left = param[3]
    length = len(attackID)
    
    times = []
    for a in attackID:
        
        print param
        graphID = param[2]
        
        if showData==True:
            print ('Attack in Progress:',a)
        
        start_time_attack = time.time()
        if a in list_net_attacks:
            edgevec,A = netcontrolz.edge_percolation_(G,a,controls=None,frac_rm_edges=param[0],num_steps=param[1])
        elif a=='betweenness':
            a = highest_betweenness
            edgevec,A = netcontrolz.edge_percolation_(G,a,controls=None,frac_rm_edges=param[0],num_steps=param[1])
        
        total_time_attack=time.time()-start_time_attack
        print total_time_attack
        
        for c in list_controls:
            profiles.append(A[c])
        profiles.append(edgevec)
        
        times.append(total_time_attack)
        avg_time = numpy.sum(times)/len(times)
        tftbc_mins = (avg_time*length*num_files_left)/60
        tftbc_hours = (avg_time*length*num_files_left)/360
        
        if showTime==True:
            print ('Total time of Attack:',total_time_attack)
            print 'Projected time for until files complete, mins:'+str(tftbc_mins)+', hours:'+str(tftbc_hours)
            print 'Percentage Complete: '+str(round((1 - ((num_files_left*length - len(times))/(length*files_o)))*100,3))+'%'
        
        if showData==True:
            names = ['Controls:','Free:','Fixed:']
            for c in range(len(list_controls)):
                print names[c]
                print A[list_controls[c]]

    numpy.savetxt(graphID + '.txt',profiles,delimiter=' ')

    return 
        





directory = raw_input('Enter directory: ')
#path = "/home/dlanigan/F1/check"
path = "/home/dlanigan/F1/" + directory

num_files_left = float(len([file for file in os.listdir(path) if file.endswith('.edgelist')]))

n = float(len([file for file in os.listdir(path) if file.endswith('.edgelist')]))

start_time1 = time.time()
for filename in os.listdir(path):       
    if filename.endswith(".edgelist"):    
        file = path + '/' + filename
        attackID = ['betweenness',netcontrolz.EDGE_ATTACK_INOUT_DEG,netcontrolz.EDGE_ATTACK_OUTIN_DEG,netcontrolz.ATTACK_RAND,netcontrolz.EDGE_ATTACK_TOTAL_DEG]
        frac_rm_edges = 0.90
        num_steps = 50
        graphID = filename.replace('.edgelist','')
        print graphID   
        G = zen.io.edgelist.read(file,directed=True,ignore_duplicate_edges=True)
        #graphID = filename.replace('.edgelist','')
        generalParam = [frac_rm_edges,num_steps,graphID,num_files_left,start_time1,n]
        #try:
        execute(G,generalParam,attackID,showData=True,showTime=True)
        num_files_left = num_files_left - 1
        #except:
            #continue

        
