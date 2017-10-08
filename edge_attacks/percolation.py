import zen
import numpy
from numpy import zeros
import netcontrolz
import networkx
import multiprocessing
import os
import time
import datetime



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


def execute_perc(frac_rm_edges = 0.90, num_steps = 50, **kwargs):
    
    """
    This can take as graph input a list of graph names, a single graph, or a directory where the graph can be obtained, or a directory where the graph names are found and a list of specific graph names to percolate
    
    The working directory is always the directory where the output files are placed. If none exists it is created.
    
    kwargs: 
    
    
    
    """
    
    keys = kwargs.keys()
    
    if kwargs is not None:
        if 'attack' in kwargs:   
            attack=kwargs.pop('attack')
        if 'show_data' in kwargs:
            show_data=kwargs.pop('show_data',True)
        if 'show_time' in kwargs:
            show_time=kwargs.pop('show_time',True)
        if 'use_file_dir' in kwargs:
            use_file_dir=kwargs.pop('use_file_dir')
        if 'use_graph_dir' in kwargs:
            use_graph_dir=kwargs.pop('use_graph_dir')
        if 'use_graph_list' in kwargs:
            use_graph_list=kwargs.pop('use_graph_list')
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    
    
    # The working directory is always the directory where the output files are going to be placed
    if 'use_file_dir' in keys:
        #check if dir exists, if not create one if so use it.
        if os.path.isdir(use_file_dir)==True:
            os.chdir(use_file_dir)
        elif os.path.isdir(use_file_dir)==False:
            os.mkdir(use_file_dir)
            os.chdir(use_file_dir)

    if ('use_file_dir' in keys)==False:
        directory = raw_input('Enter File Directory: ')
        
        cdw = os.getcwd()
        path = cdw + '/' + directory
        
        if os.path.isdir(path)==True:
            os.chdir(path)
        elif os.path.isdir(path)==False:
            os.mkdir(path)
            os.chdir(path)
    
    
    list_net_attacks = ['betweenness',netcontrolz.EDGE_ATTACK_INOUT_DEG,netcontrolz.EDGE_ATTACK_OUTIN_DEG,netcontrolz.ATTACK_RAND,netcontrolz.EDGE_ATTACK_TOTAL_DEG]
    list_controls = [netcontrolz.CONTROL_ROBUSTNESS,netcontrolz.REACHABILITY_ROBUSTNESS_FIXED,netcontrolz.REACHABILITY_ROBUSTNESS_FREE]                 
                     
    #attackID = ['betweenness',netcontrolz.EDGE_ATTACK_INOUT_DEG,netcontrolz.EDGE_ATTACK_OUTIN_DEG,netcontrolz.ATTACK_RAND,netcontrolz.EDGE_ATTACK_TOTAL_DEG]
     
    #get the graphs to percolate
    graphs = []
    if 'use_graph_dir' in keys:
        if ('use_file_list'in keys)==False:
            for filename in os.listdir(use_graph_dir):       
                if filename.endswith(".edgelist"):    
                    file = use_graph_dir + '/' + filename
                    G = zen.io.edgelist.read(file,directed=True,ignore_duplicate_edges=True)
                    graphs.append((G,filename.replace('.edgelist','')))
        
        elif 'use_file_list' in keys:
            for filename in use_file_list:       
                if filename.endswith(".edgelist"):    
                    file = use_graph_dir + '/' + filename
                    G = zen.io.edgelist.read(file,directed=True,ignore_duplicate_edges=True)
                    graphs.append((G,filename.replace('.edgelist','')))
                
                elif filename.endswith(".edgelist")==False:    
                    filename = filename + ".edgelist"
                    file = use_graph_dir + '/' + filename
                    G = zen.io.edgelist.read(file,directed=True,ignore_duplicate_edges=True)
                    graphs.append((G,filename.replace('.edgelist','')))
    
    if ('use_graph_dir' in keys)==False:
        if ('use_file_list'in keys)==False:
            for filename in os.listdir(os.getcwd()):       
                if filename.endswith(".edgelist"):    
                    file = use_graph_dir + '/' + filename
                    graphID = filename.replace('.edgelist','')
                    G = zen.io.edgelist.read(file,directed=True,ignore_duplicate_edges=True)
                    graphs.append((G,filename.replace('.edgelist','')))
                    
        elif 'use_file_list' in keys:
            for filename in use_file_list:       
                if filename.endswith(".edgelist"):    
                    file = use_graph_dir + '/' + filename
                    graphID = filename.replace('.edgelist','')
                    G = zen.io.edgelist.read(file,directed=True,ignore_duplicate_edges=True)
                    graphs.append((G,filename.replace('.edgelist','')))
                    
                elif (filename.endswith(".edgelist"))==False:    
                    filename = filename + ".edgelist"
                    file = use_graph_dir + '/' + filename
                    graphID = filename.replace('.edgelist','')
                    G = zen.io.edgelist.read(file,directed=True,ignore_duplicate_edges=True)
                    graphs.append((G,filename.replace('.edgelist','')))
                     
                        
    num_attacks = len(attack)
    num_graphs = len(graphs)
    
    num_files = float(len([file for file in os.listdir(os.getcwd()) if file.endswith('.txt')]))
    num_files_o = num_files
    start_time1 = time.time()
                     
    profiles=[]                
    times_graph=[]
    
    for G in graphs:
        time1_graph_start = time.time()
        times_attack=[]
        for a in attack:
            
            start_time_attack = time.time()
            if show_data==True:
                print ('Attack in Progress:', a)

            start_time_attack = time.time()
            if a in list_net_attacks:
                edgevec,A = netcontrolz.edge_percolation_(G[0],a,controls=None,frac_rm_edges=frac_rm_edges,num_steps=num_steps)
            elif a=='betweenness':
                a = highest_betweenness
                edgevec,A = netcontrolz.edge_percolation_(G[0],a,controls=None,frac_rm_edges=frac_rm_edges,num_steps=num_steps)

            tot_t_attack=time.time()-start_time_attack

            for c in list_controls:
                profiles.append(A[c])
            profiles.append(edgevec)

            times_attack.append(tot_t_attack)
            
            avg_t_attack = numpy.sum(times_attack)/(len(times_attack)*60)
            
            if len(times_graph)>0:
                avg_t_graph = numpy.sum(times_graph)/(len(times_graph)*60)
            else:
                avg_t_graph = 0
                
            avg = (avg_t_graph + num_attacks*avg_t_attack)/2.0
            
            print avg
            t_total = int((num_graphs - num_files)*avg)
            print 'this is t_total:' + str(t_total)
            
            done_time = datetime.datetime.now() + datetime.timedelta(minutes=t_total)
                     
            if show_time==True:
                print ('Total time of Attack in minutes:',tot_t_attack/60.0)
                print 'Percentage Complete: '+str(round((1 - ((( num_graphs - num_files )*num_attacks - len(times_attack))/(num_attacks*num_graphs)))*100,3))+'%'
                print 'Projected time for all graphs to complete: ' + str(done_time)
            if show_data==True:
                names = ['Controls:','Free:','Fixed:']
                for c in range(len(list_controls)):
                    print names[c]
                    print A[list_controls[c]]

        
        numpy.savetxt(G[1] + '.txt',profiles,delimiter=' ')
        time1_graph_end = time.time()
        times_graph.append(time1_graph_end - time1_graph_start)
        num_files = float(len([file for file in os.listdir(os.getcwd()) if file.endswith('.txt')]))
                     
        
    return 