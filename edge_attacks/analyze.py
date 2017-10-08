import zen, sys, numpy, os
import random, decimal, matplotlib
import matplotlib.pyplot as plt
sys.path.append('C:\Users\David\Documents\UTD Graduate\Semester I\Dynamics of Complex Networks\Labs\zend3js')
import d3js
import networkx as nx
import igraph
import time
from sklearn import mixture
import numpy as np
import matplotlib.mlab as mlab
from scipy.stats import norm
import math
#import label_rank

#if __name__=='__main__':

def highest_betweenness(G):

    """

    Returns the edge index that has the highest betweeness centrality.

    Currently we rely on the netowrkx to calculate this statistic.

    """

    Gnx = zen.nx.to_networkx(G)

    edge_btw = nx.edge_betweenness_centrality(Gnx,normalized = False)

    high_btw = sorted(edge_btw.items(), key=lambda x: x[1],reverse = True)[::-1]

    #eidx = G.edge_idx(high_btw_endpts[0],high_btw_endpts[1])

    return high_btw

def handle_keys(kwargs):
    keys = kwargs.keys()
    
    print keys
    
    if kwargs is not None:
        if 'show' in kwargs:
            show=kwargs.pop('show',True)
        if 'use_file_dir' in kwargs:
            use_file_dir=kwargs.pop('use_file_dir')
        if 'use_graph_dir' in kwargs:
            use_graph_dir=kwargs.pop('use_graph_dir')
        if 'use_graph_list' in kwargs:
            use_graph_list=kwargs.pop('use_graph_list')
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    
    
    if 'use_file_dir' in keys:
        os.chdir(use_file_dir)
    
    """
    if 'use_file_dir' in keys==False:
        directory = raw_input('Enter File Directory: ')
        
        cdw = os.getcwd()
        path = cdw + '/' + directory
        os.chdir(path)
    """
                    
                     
    #attackID = ['betweenness',netcontrolz.EDGE_ATTACK_INOUT_DEG,netcontrolz.EDGE_ATTACK_OUTIN_DEG,netcontrolz.ATTACK_RAND,netcontrolz.EDGE_ATTACK_TOTAL_DEG]
     
    #get files
    graphs = []
    if 'use_graph_dir' in keys:
        if ('use_file_list'in keys)==False:
            print 'here'
            for filename in os.listdir(use_graph_dir):       
                if filename.endswith(".edgelist"):
                    graphs.append( filename.replace('.edgelist','') )

        elif 'use_file_list' in keys:
            for filename in use_file_list:       
                if filename.endswith(".edgelist"):    
                    graphs.append( filename.replace('.edgelist','') )
                elif filename.endswith(".edgelist")==False:    
                    graphs.append( filename )
    
    
    if 'use_graph_dir' in keys==False:
        if ('use_file_list'in keys)==False:
            for filename in os.listdir(os.getcwd()):       
                if filename.endswith(".edgelist"):    
                    graphs.append( filename.replace('.edgelist','') )
        elif 'use_file_list' in keys:
            for filename in use_file_list:       
                if filename.endswith(".edgelist"):    
                    graphs.append( filename.replace('.edgelist','') )
                elif filename.endswith(".edgelist")==False:    
                    graphs.append( filename )
                    
    return
    
def view_net(G,nx_graph=False):

    if nx_graph==True:
        G = zen.nx.from_networkx(G)

    d3 = d3js.D3jsRenderer(G, canvas_size=(1000,2000), interactive=True, autolaunch=False)

    return


def get_graphs_number(param,**kwargs):

    graph_type=kwargs.pop('graph_type')

    counter = 0
    for i in range(len(param)):

        if graph_type == 'dd':
            G = zen.generating.duplication_divergence_iky(nodes,param[i],directed=True)
        elif graph_type == 'ba':
            G = zen.generating.barabasi_albert(nodes,param[i],directed=True)
        elif graph_type == 'er':
            G = zen.generating.erdos_renyi(nodes,param[i],directed=True)
        elif graph_type == 'la':
            G = zen.generating.local_attachment(nodes,param[i],3)


        prof = zen.control.profile(G,normalized=True)
        print ('control profile:', (prof,i))
        print ('degree:',len(G.edges_())/len(G.nodes_()))
        print len(G.nodes_())
        print len(G.edges_())

        if counter==0:

            if any(File.endswith(".edgelist") for File in os.listdir(os.getcwd()))==True:
                for file in os.listdir(os.getcwd()):
                    if file.endswith(".edgelist"):
                        max = 0
                        if int(file[8])>=max:
                            max = int(file[8])
                            counter = counter + 1
            elif any(File.endswith(".edgelist") for File in os.listdir(os.getcwd()))==False:
                max = -1
                counter = counter + 1

            filename = 'network_'+str(max+1) +'_'+str(i)+'.edgelist'
            zen.io.edgelist.write(G,filename)

        elif counter >0:
            filename = 'network_'+str(max+1) +'_'+str(i)+'.edgelist'
            zen.io.edgelist.write(G,filename)

    return

def get_graphs_range(amount=5,nodes=1000,commy_dist=[50,50,50,50],param_range=(25,5),param_range2=(3,7),**kwargs):

    if kwargs is not None:
        if 'graph_type' in kwargs:   
            graph_type=kwargs.pop('graph_type')
        if 'use_dir' in kwargs:
            use_dir=kwargs.pop('use_dir')
        if 'make_dir' in kwargs:
            make_dir=kwargs.pop('make_dir')
        if 'names' in kwargs:
            names=kwargs.pop('names')
            
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    #if directory is given set cwd to that directory
    if use_dir:
        #check if dir exists, if not create one if so use it.
        if os.path.isdir(use_dir)==True:
            os.chdir(use_dir)
        elif os.path.isdir(use_dir)==False:
            os.mkdir(use_dir)
            os.chdir(use_dir)
    
    if param_range:# in globals():
        low=param_range[0]
        high=param_range[1]
    if param_range2:# in locals() or param_range2 in globals():
        low2=param_range2[0]
        high2=param_range2[1]


    #check for files in the directory
    amount_files_o = len([file for file in os.listdir(os.getcwd()) if file.endswith('.edgelist')])
    amount_files=0
    i=0
    counter = 0
    while (amount_files < amount_files_o + amount):

        amount_files = len([file for file in os.listdir(os.getcwd()) if file.endswith('.edgelist')])
        i=i + 1

        if graph_type == 'dd':
            l = float(decimal.Decimal(random.randrange(low, high))/100)
            G = zen.generating.duplication_divergence_iky(nodes,l,directed=True)
        elif graph_type == 'ba':
            l = random.randrange(low,high)
            G = zen.generating.barabasi_albert(nodes,40,directed=True)
        elif graph_type == 'er':
            l = float(decimal.Decimal(random.randrange(low, high))/100)
            G = zen.generating.rgm.erdos_renyi(nodes,0.05,directed=True,self_loops=False)
        elif graph_type == 'la':
            l = random.randrange(low, high)
            r = random.randrange(1, l)
            G = zen.generating.local_attachment(nodes,l,r)
        elif graph_type == 'commy_rand':
            cV = commy_dist
            l = float(decimal.Decimal(random.randrange(low, high))/10000)
            l2 = float(decimal.Decimal(random.randrange(low2, high2))/10000)
            G = nx.random_partition_graph(cV,l,l2,directed=True)
            G = zen.nx.from_networkx(G)
        #elif graph_type=='commy_gauss':
            #G = nx.gaussian_random_partition_graph(100,10,10,.25,.1)

        prof = zen.control.profile(G,normalized=True)
        print ('control profile:', (prof,i))
        print ('degree:',len(G.edges())/len(G.nodes()))
        print len(G.nodes_())
        print len(G.edges_())
        
        
        if names=='params':
            
            #if graph_type=='commy_gauss':
                #code
            
            if graph_type=='commy_rand':

                filename = str(graph_type)+'_'+ str(commy_dist)+'_'+str(l) +'_'+str(l2)+'_' +str(time.time()) +'.edgelist'

            
            zen.io.edgelist.write(G,filename)
        
        if names=='generic':

            if counter==0:
                if any(File.endswith(".edgelist") for File in os.listdir(os.getcwd()))==True:
                    for file in os.listdir(os.getcwd()):
                        if file.endswith(".edgelist"):
                            max = 0
                            if file[9]=='_' and int(file[8])>=max:
                                max = int(file[8])
                            elif int(file[8] + file[9])>=max:
                                counter = counter + 1
                elif any(File.endswith(".edgelist") for File in os.listdir(os.getcwd()))==False:
                    max = -1
                    counter = counter + 1

                    filename = str(graph_type)+'_'+str(max+1) +'_'+str(i)+'.edgelist'
                zen.io.edgelist.write(G,filename)

            elif counter >0:
                if graph_type=='commy_rand':
                    #filename = str(graph_type)+'_'+ len(commy_dist)+'_'+str(max+1) +'_'+str(i)+'.edgelist'
                    filename = str(graph_type)+'_'+ str(len(commy_dist))+'_'+str(l) +'_'+str(l2)+'.edgelist'
                else:
                    filename = str(graph_type)+'_'+str(max+1) +'_'+str(i)+'.edgelist'

                zen.io.edgelist.write(G,filename)

        ###Code for Control profile integration
        """
        if type_prof=='sd':
            prof_space=0
        if type_prof=='edd':
            prof_space=1
        if type_prof=='idd':
            prof_space=2


        if prof[prof_space]>0.89:
            if len(G.edges_())/len(G.nodes_()) >40 and len(G.edges_())/len(G.nodes_()) <= 55:
                if counter==0:

                    if any(File.endswith(".edgelist") for File in os.listdir(os.getcwd()))==True:
                        for file in os.listdir(os.getcwd()):
                            if file.endswith(".edgelist"):
                                max = 0
                                if int(file[8])>=max:
                                    max = int(file[8])
                                    counter = counter + 1
                    elif any(File.endswith(".edgelist") for File in os.listdir(os.getcwd()))==False:
                        max = -1
                        counter = counter + 1

                    #filename = 'network_'+str(max+1) +'_'+str(i)+'.edgelist'
                    filename = 'network_'+str(max+1) +'_'+str(i)+'.edgelist'
                    zen.io.edgelist.write(G,filename)

                elif counter >0:
                    filename = 'network_'+str(max+1) +'_'+str(i)+'.edgelist'
                    zen.io.edgelist.write(G,filename)

        """
    return


def print_net_props(*G,**kwargs):

    
    def get_av_degrees(ddist):

        degrees = []
        for i in range(len(ddist)):
            degrees.append(i*ddist[i])

        average = numpy.sum(degrees)/numpy.sum(ddist)

        return average
    
    
    keys = kwargs.keys()

    if kwargs is not None:
        if 'graph_path_given' in kwargs:
            graph_path_given=kwargs.pop('graph_path_given',True)
        if 'use_file_dir' in kwargs:
            use_file_dir=kwargs.pop('use_file_dir')
        if 'use_graph_dir' in kwargs:
            use_graph_dir=kwargs.pop('use_graph_dir')
        if 'use_graph_list' in kwargs:
            use_graph_list=kwargs.pop('use_graph_list')
        if 'use_file_list' in kwargs:
            use_file_list=kwargs.pop('use_file_list')
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)



    if graph_path_given==True:
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

    elif graph_path_given==False:
        graphs = [(G,'given graph')]
    

    
    for g in graphs:
        
        G = g[0]
        
        ddist_in = zen.degree.ddist(G,direction='in_dir',normalize=True)
        ddist_out = zen.degree.ddist(G,direction='out_dir',normalize=True)
        ddist = zen.degree.ddist(G,normalize=True)

        print 'The number of vertices for the ' +  network + ' network is:'
        print len(G.nodes())
        #print '\n'
        print 'The number of edges for the ' + network + ' network is:'
        print len(G.edges())
        #print '\n'

        print 'The average degree of the '+ network+ ' network is:'
        print get_av_degrees(ddist)

        print 'The average out degree of the '+ network+ ' network is:'
        print get_av_degrees(ddist_out)

        print 'The average in degree of the '+ network+ ' network is:'
        print get_av_degrees(ddist_in)

        



    return



def create_figs(size=15,**kwargs):
    
    keys = kwargs.keys()
    
    print keys
    
    if kwargs is not None:
        if 'show' in kwargs:
            show=kwargs.pop('show',True)
        if 'use_file_dir' in kwargs:
            use_file_dir=kwargs.pop('use_file_dir')
        if 'use_graph_dir' in kwargs:
            use_graph_dir=kwargs.pop('use_graph_dir')
        if 'use_graph_list' in kwargs:
            use_graph_list=kwargs.pop('use_graph_list')
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    
    
    if 'use_file_dir' in keys:
        os.chdir(use_file_dir)
    
    """
    if 'use_file_dir' in keys==False:
        directory = raw_input('Enter File Directory: ')
        
        cdw = os.getcwd()
        path = cdw + '/' + directory
        os.chdir(path)
    """
                    
                     
    #attackID = ['betweenness',netcontrolz.EDGE_ATTACK_INOUT_DEG,netcontrolz.EDGE_ATTACK_OUTIN_DEG,netcontrolz.ATTACK_RAND,netcontrolz.EDGE_ATTACK_TOTAL_DEG]
     
    #get files
    graphs = []
    if 'use_graph_dir' in keys:
        if ('use_file_list'in keys)==False:
            print 'here'
            for filename in os.listdir(use_graph_dir):       
                if filename.endswith(".edgelist"):
                    graphs.append( filename.replace('.edgelist','') )

        elif 'use_file_list' in keys:
            for filename in use_file_list:       
                if filename.endswith(".edgelist"):    
                    graphs.append( filename.replace('.edgelist','') )
                elif filename.endswith(".edgelist")==False:    
                    graphs.append( filename )
    
    
    if 'use_graph_dir' in keys==False:
        if ('use_file_list'in keys)==False:
            for filename in os.listdir(os.getcwd()):       
                if filename.endswith(".edgelist"):    
                    graphs.append( filename.replace('.edgelist','') )
        elif 'use_file_list' in keys:
            for filename in use_file_list:       
                if filename.endswith(".edgelist"):    
                    graphs.append( filename.replace('.edgelist','') )
                elif filename.endswith(".edgelist")==False:    
                    graphs.append( filename )
    
    
    print os.getcwd()
    
    for g in graphs:

        try:
            A = numpy.loadtxt(g + '.txt')
        
        except IOError:
            continue
        #ORDER OF ATTACK:
        attackID = ['betweenness','inout','outin','rand','total']
        controlID = ['controls','reach_fixed','reach_fixed']

        n = 4
        E_idx1 = 3
        E_idx2 = E_idx1 + n
        E_idx3 = E_idx2 + n
        E_idx4 = E_idx3 + n
        E_idx5 = E_idx4 + n

        idx1 = 0
        idx2 = n
        idx3 = idx2 + n
        idx4 = idx3 + n
        idx5 = idx3 + n

        color = ['b','g','r','c','y']
        
        for i in range(len(controlID)):
            try:
                plt.scatter(A[E_idx1],A[idx1 + i]/A[idx1 + 1][0],label=attackID[0],s=size,color=color[0])#,alpha=alpha)
                plt.scatter(A[E_idx2],A[idx2 + i]/A[idx2 + 1][0],label=attackID[1],s=size,color=color[1])#,alpha=alpha)
                plt.scatter(A[E_idx3],A[idx3 + i]/A[idx3 + 1][0],label=attackID[2],s=size,color=color[2])#,alpha=alpha)
                plt.scatter(A[E_idx4],A[idx4 + i]/A[idx4 + 1][0],label=attackID[3],s=size,color=color[3])#,alpha=alpha)
                plt.scatter(A[E_idx5],A[idx5 + i]/A[idx5 + 1][0],label=attackID[4],s=size,color=color[4])#,alpha=alpha)
            except IndexError:
                continue
            
            
            #"""
            plt.xlabel('$l/L$')
            plt.ylabel('$n/N$')
            plt.ylim([0,1.1])
            plt.legend()
            plt.title('Graph: ' + g + ', ' + 'Control Type: ' + controlID[i])
            
            if show==True:
                plt.show()
            if show==False:
                fig.savefig(graphID + '_' + i + '.png')   
    
    return 

def anylz_histo(bins1=100,bins2=200,*G, **kwargs):
    
    keys = kwargs.keys()

    if kwargs is not None:
        if 'show_stats' in kwargs:
            show_stats=kwargs.pop('show_stats',True)
        if 'show_hist' in kwargs:
            show_hist=kwargs.pop('show_hist',True)
        if 'use_file_dir' in kwargs:
            use_file_dir=kwargs.pop('use_file_dir')
        if 'use_graph_dir' in kwargs:
            use_graph_dir=kwargs.pop('use_graph_dir')
        if 'use_graph_list' in kwargs:
            use_graph_list=kwargs.pop('use_graph_list')
        if 'use_file_list' in kwargs:
            use_file_list=kwargs.pop('use_file_list')
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)

    graphs = []
    if G:
        graphs.append(G)
        
    if 'use_graph_dir' in keys:
        if ('use_file_list'in keys)==False:
            for filename in os.listdir(use_graph_dir):       
                if filename.endswith(".edgelist"):
                    graphs.append( use_graph_dir + '/' +filename )
        elif ('use_file_list'in keys)==True:
            for filename in use_file_list:
                    graphs.append( use_graph_dir + '/' + filename )
        
        
    for g in graphs:
        G = zen.io.edgelist.read(g,ignore_duplicate_edges=True)
        z = highest_betweenness(G)
        l = []
        for i in z:
            l.append(i[1])
        
        z = numpy.zeros((len(l),1))
        for i in range(len(z)):
            z[i] = l[i]

        z = numpy.zeros((len(l),1))

        for i in range(len(z)):
            z[i] = l[i]


        gmix = mixture.GaussianMixture(n_components=2,covariance_type='full').fit(z)

        c=gmix.predict(z)
        c = list(c)

        z = z.reshape(( numpy.shape(z)[1],numpy.shape(z)[0] )).tolist()[0]


        g1=[]
        g2=[]

        for i in range(len(c)):
            if c[i]==0:
                g1.append(z[i])
            elif c[i]==1:
                g2.append(z[i])

        mu1 = numpy.mean(g1)
        mu2 = numpy.mean(g2)

        sigma1 = numpy.std(g1)
        sigma2 = numpy.std(g2)

        if show_stats==True:
            #Ashmans D
            D = abs((math.pow(2,0.5))*(mu2-mu1))/(math.sqrt(math.pow(sigma1,2) + math.pow(sigma2,2)))

            #Bimodal seperation
            S = (mu2 - mu1)/2*(sigma1 + sigma2)

            #print ("Amp Ratio", Ar)
            print ("Bimodal Seperation", S)
            print ("Ashmans D", D)
            print ( "Crossover Should Appear at", float(len(g2))/float(len(z)) )


        if show_hist==True:
            dist1=norm(mu1,sigma1) 
            dist2=norm(mu2,sigma2) 
            plt.hist(g1,bins=bins1 ,color='b')
            plt.hist(g2,bins=bins2 ,color='g')
            plt.show()

    
    return
    
def betw_histo(bins=200, *G, **kwargs):

    keys = kwargs.keys()
    if kwargs is not None:
        if 'show_stats' in kwargs:
            show_stats=kwargs.pop('show_stats',True)
        if 'show_hist' in kwargs:
            show_hist=kwargs.pop('show_hist',True)
        if 'use_file_dir' in kwargs:
            use_file_dir=kwargs.pop('use_file_dir')
        if 'use_graph_dir' in kwargs:
            use_graph_dir=kwargs.pop('use_graph_dir')
        if 'use_graph_list' in kwargs:
            use_graph_list=kwargs.pop('use_graph_list')
        if 'use_file_list' in kwargs:
            use_file_list=kwargs.pop('use_file_list')
    if kwargs:
        raise TypeError('Unexpected **kwargs: %r' % kwargs)
    
    
    
    graphs = []
    
    if G: 
        graphs.append(G)
        
    if 'use_graph_dir' in keys:
        if ('use_file_list'in keys)==False:
            for filename in os.listdir(use_graph_dir):       
                if filename.endswith(".edgelist"):
                    graphs.append( use_graph_dir + '/' +filename )
        elif ('use_file_list'in keys)==True:
            for filename in use_file_list:
                    graphs.append( use_graph_dir + '/' + filename )
        
        
    for g in graphs:
        G = zen.io.edgelist.read(g,ignore_duplicate_edges=True)
        z = highest_betweenness(G)
        l = []
        for i in z:
            l.append(i[1])
        z = numpy.zeros((len(l),1))
        for i in range(len(z)):
            z[i] = l[i]

        plt.hist(z,bins=bins)
        plt.title(g)
        plt.show()
    
    return
    
    