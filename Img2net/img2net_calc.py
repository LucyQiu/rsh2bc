##################################################################### imports

import matplotlib
import matplotlib.pyplot as plt
import multiprocessing
import networkx as nx
import numpy as np
import os
import scipy as sp
import scipy.ndimage
from mpl_toolkits.mplot3d import Axes3D

import img2net_help

##################################################################### img2net

def calc(self,gtk,dir_input,gridtype,nullmodel,R,lz,dx,dy,dz,pbx,pby,pbz,vax,vay,cores):

#    dir_input='/media/breuer/Data/2014_img2net_TestData/3D'
#    dir_input='C:/Users/David/Dropbox/2013_CytoQuant/img2net_TestData/3D'
#    gridtype='rectangular'#'hexagonal'#'triangular'#
#    nullmodel='edges'
#    R=4
#    dx=dy=dz=60#40#20#
#    lz=4#1#
#    vax,vay=4,4
#    pbx,pby,pbz=0,0,0
#    cores=3

    ############################################# initialize parameters

    #server=pp.Server(cores)
    #multiprocessing.freeze_support()
    core=multiprocessing.cpu_count()
    if(cores>core):
        cores=core
        message = gtk.MessageDialog(parent=self.window,type=gtk.MESSAGE_INFO, buttons=gtk.BUTTONS_YES_NO,message_format='More cores chosen than available.')
        message.format_secondary_text('Cores available: '+str(cores)+'\n\nProceed with '+str(cores)+' cores?')
        result=message.run()
        message.destroy()
        if result == gtk.RESPONSE_NO:
            return 0

    pool=multiprocessing.Pool(processes=cores)

    dir_input+='/'
    treatments=np.sort(os.listdir(dir_input))
    treatments=[t for t in treatments if not 'Output_' in t]
    T=len(treatments)
    experiments=[np.sort(os.listdir(dir_input+treatments[t])) for t in range(T)]
    E=[len(experiments[t]) for t in range(T)]
    images=[[np.sort(os.listdir(dir_input+treatments[t]+'/'+experiments[t][e])) for e in range(E[t])] for t in range(T)]
    I=[[len(images[t][e])/lz for e in range(E[t])] for t in range(T)]

    name='gridtype='+str(gridtype)+'_nullmodel='+str(nullmodel)+'_R='+str(R)+'_dx='+str(dx)+'_dy='+str(dy)+'_dz='+str(dz)+'_vax='+str(vax)+'_vay='+str(vay)+'_lz='+str(lz)+'_pbx='+str(pbx)+'_pby='+str(pby)+'_pbz='+str(pbz)+'/'
    dir_output=dir_input+'Output_'+name
    subfolders=['data_posi','data_conv','data_grph','data_datn','data_datr','data_prop','data_readable','plot_grid','plot_time','plot_dist']

    ############################################# generate folders and check: overwrite?

    if(not os.path.isdir(dir_output)):
        os.makedirs(dir_output)
    else:
        message = gtk.MessageDialog(parent=self.window,type=gtk.MESSAGE_INFO, buttons=gtk.BUTTONS_YES_NO,message_format='Output folder exists. Content will be overwritten.')
        message.format_secondary_text('Path: '+dir_output+'\n\nProceed and overwrite?')
        result=message.run()
        message.destroy()
        if result == gtk.RESPONSE_NO:
            return 0

    for subfolder in subfolders:
        if(not os.path.isdir(dir_output+subfolder)):
                os.makedirs(dir_output+subfolder)

    if(os.path.isfile(dir_output+'data_readable/data_readable.txt')):
        os.remove(dir_output+'data_readable/data_readable.txt')

    ############################################# check for errors in: file types, image sizes, graph sizes

    for t in range(T):
        for e in range(E[t]):
            dir_te=dir_input+treatments[t]+'/'+experiments[t][e]
            dir_sh=dir_te+'/'+images[t][e][0]
            for i in range(I[t][e]):
                dir_err=dir_input+treatments[t]+'/'+experiments[t][e]+'/'+images[t][e][i]

                if(not images[t][e][i].lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))):
                    message = gtk.MessageDialog(parent=self.window,type=gtk.MESSAGE_INFO, buttons=gtk.BUTTONS_OK,message_format='Wrong image format.')
                    message.format_secondary_text('Path: '+dir_err+'\n\nRequired: jpg, tiff, or png.\nFound: '+images[t][e][i].split('.')[-1])
                    result=message.run()
                    message.destroy()
                    if result == gtk.RESPONSE_OK:
                        return 0

                if(i==0):
                    sh=np.shape(sp.ndimage.imread(dir_sh))[0:2]
                shi=np.shape(sp.ndimage.imread(dir_err))[0:2]

                if(sh!=shi):
                    message = gtk.MessageDialog(parent=self.window,type=gtk.MESSAGE_INFO, buttons=gtk.BUTTONS_OK,message_format='Wrong image size.')
                    suffix=': '
                    if(images[t][e][i].lower().endswith(('.tif','.tiff'))):
                        suffix='(try 8-bit): '
                    message.format_secondary_text('Path: '+dir_te+'\n\nRequired'+suffix+str(sh)+'.\nFound: '+str(shi))
                    result=message.run()
                    message.destroy()
                    if result == gtk.RESPONSE_OK:
                        return 0

            N,nnx,nny,nnz,posi,L,edges,convs=img2net_help.grid_grid(gridtype,shi[1],shi[0],lz,dx,dy,dz,pbx,pby,pbz,vax,vay,0,'','')

            if(N==0 or L==0):
                message = gtk.MessageDialog(parent=self.window,type=gtk.MESSAGE_INFO, buttons=gtk.BUTTONS_OK,message_format='Graph contains no nodes.')
                message.format_secondary_text('Path: '+dir_te+'\n\nCheck image sizes or try smaller grid constants.')
                result=message.run()
                message.destroy()
                if result == gtk.RESPONSE_OK:
                    return 0

    ############################################# program

    #temp=time.time()

    c=1.0
    C=1.0*np.nansum(E)+1

    for t in range(T):

        for e in range(E[t]):

            dir_te=dir_input+treatments[t]+'/'+experiments[t][e]

            ################################################################# grid

            print('t',t+1,T,'e',e+1,E[t],'grid')

            s=1.0
            S=I[t][e]*(2.0+R)+2

            self.builder.get_object('progressbar1').set_text('T: '+str(t+1)+'/'+str(T)+', E: '+str(e+1)+'/'+str(E[t]))
            self.builder.get_object('progressbar2').set_text('initializing data')

            dir_te=dir_input+treatments[t]+'/'+experiments[t][e]+'/'
            ims=sp.ndimage.imread(dir_te+images[t][e][0])
            sh=np.shape(ims)
            lx,ly,lz=sh[1],sh[0],lz
            dir_posi=dir_output+'data_posi/data_posi_T='+str(t).zfill(4)+'_E='+str(e).zfill(4)
            dir_conv=dir_output+'data_conv/data_conv_T='+str(t).zfill(4)+'_E='+str(e).zfill(4)
            N,nnx,nny,nnz,posi,L,edges,convs=img2net_help.grid_grid(gridtype,lx,ly,lz,dx,dy,dz,pbx,pby,pbz,vax,vay,1,dir_posi,dir_conv)

            self.builder.get_object('progressbar2').set_fraction(s/S)
            s+=1
            while gtk.events_pending():
                gtk.main_iteration()

            ################################################################# graph

            print('t',t+1,T,'e',e+1,E[t],'graph')

            self.builder.get_object('progressbar2').set_text('constructing graphs')

            jobs=[]
            graphn=[[] for i in range(I[t][e])]
            for i in range(I[t][e]):
                inp=[t,e,i,dir_te,images[t][e][lz*i:lz*i+lz],L,edges,dir_posi,dir_conv,dz]
                # TODO uncomment for multiprocessing
                #jobs.append(pool.apply_async(img2net_help.grid_all,args=(inp,)))
            #for job in jobs:
                #res=job.get()
                res=img2net_help.grid_all(inp)
                graphn[res[2]]=res[3]
                name=dir_output+'data_grph/data_grph_T='+str(res[0]).zfill(4)+'_E='+str(res[1]).zfill(4)+'_I='+str(res[2]).zfill(4)
                np.save(name,list(nx.to_edgelist(res[3])))

                self.builder.get_object('progressbar2').set_fraction(s/S)
                s+=1
                while gtk.events_pending():
                    gtk.main_iteration()

            ################################################################# observed network properties

            print('t',t+1,T,'e',e+1,E[t],'obs network')

            self.builder.get_object('progressbar2').set_text('computing properties')

            jobs=[]
            for i in range(I[t][e]):
                inp=[t,e,i,99,graphn[i],posi]
                # TODO uncomment for multiprocessing
               # jobs.append(pool.apply_async(img2net_help.graph_all,args=(inp,)))
            #for job in jobs:
                #res=job.get()
                res=img2net_help.graph_all(inp)
                np.save(dir_output+'data_datn/data_datn_T='+str(res[0]).zfill(4)+'_E='+str(res[1]).zfill(4)+'_I='+str(res[2]).zfill(4),res[4])

                self.builder.get_object('progressbar2').set_fraction(s/S)
                s+=1
                while gtk.events_pending():
                    gtk.main_iteration()

            labels=res[5]
            np.save(dir_output+'data_prop/data_prop_T='+str(t).zfill(4)+'_E='+str(e).zfill(4),labels)

            ################################################################# null network properties

            print('t',t+1,T,'e',e+1,E[t],'null network')

            self.builder.get_object('progressbar2').set_text('evaluating null model')

            jobs=[]
            for i in range(I[t][e]):
                for r in range(R):
                    inp=[t,e,i,r,img2net_help.graph_null(graphn[i],nnx,nny,nullmodel),posi]
                    # TODO uncomment for multiprocessing
                    #jobs.append(pool.apply_async(img2net_help.graph_all,args=(inp,)))
            #for job in jobs:
                #res=job.get()
                    res=img2net_help.graph_all(inp)
                    np.save(dir_output+'data_datr/data_datr_T='+str(res[0]).zfill(4)+'_E='+str(res[1]).zfill(4)+'_I='+str(res[2]).zfill(4)+'_R='+str(res[3]).zfill(4),res[4])

                    self.builder.get_object('progressbar2').set_fraction(s/S)
                    s+=1
                    while gtk.events_pending():
                        gtk.main_iteration()


            self.builder.get_object('progressbar1').set_fraction(c/C)
            c+=1
            while gtk.events_pending():
                gtk.main_iteration()

    print('s',s,S,'c',c,C)

    #print(time.time()-temp)

    #return 0

################################################################# save data

    s=1.0
    S=1.0*np.nansum(E)+len(labels)+3.0

    self.builder.get_object('progressbar1').set_text('generating output')
    self.builder.get_object('progressbar1').set_fraction(1.0)
    self.builder.get_object('progressbar2').set_text('saving data')
    self.builder.get_object('progressbar2').set_fraction(s/S)
    while gtk.events_pending():
        gtk.main_iteration()

    labels=np.load(dir_output+'data_prop/data_prop_T='+str(0).zfill(4)+'_E='+str(0).zfill(4)+'.npy')

    with open(dir_output+'data_readable/data_readable.txt',"a") as out:
        out.write('\t'.join([str(a) for a in np.hstack([['treatment','experiment','data type','image'],labels])]))
        out.write('\n')
        for t in range(T):
            for e in range(E[t]):
                for i in range(I[t][e]):
                    datn=np.load(dir_output+'data_datn/data_datn_T='+str(t).zfill(4)+'_E='+str(e).zfill(4)+'_I='+str(i).zfill(4)+'.npy')
                    out.write('\t'.join([str(a) for a in np.hstack([[treatments[t],experiments[t][e],'observed',i],datn])]))
                    out.write('\n')
                    for r in range(R):
                        datr=np.load(dir_output+'data_datr/data_datr_T='+str(t).zfill(4)+'_E='+str(e).zfill(4)+'_I='+str(i).zfill(4)+'_R='+str(r).zfill(4)+'.npy')
                        out.write('\t'.join([str(a) for a in np.hstack([[treatments[t],experiments[t][e],'null',i],datr])]))
                        out.write('\n')

    ################################################################# plot boxplots

    matplotlib.rcParams.update({'font.size': 15})

    s+=1
    self.builder.get_object('progressbar2').set_text('plotting data')
    self.builder.get_object('progressbar2').set_fraction(s/S)
    while gtk.events_pending():
        gtk.main_iteration()

    for l,label in enumerate(labels):

        print('label',l+1,len(labels))

        dn=[[] for t in range(T)]
        dd=[[] for t in range(T)]
        for t in range(T):
            for e in range(E[t]):
                for i in range(I[t][e]):
                    datn=np.load(dir_output+'data_datn/data_datn_T='+str(t).zfill(4)+'_E='+str(e).zfill(4)+'_I='+str(i).zfill(4)+'.npy')
                    dn[t].append(datn[l])
                    for r in range(R):
                        datr=np.load(dir_output+'data_datr/data_datr_T='+str(t).zfill(4)+'_E='+str(e).zfill(4)+'_I='+str(i).zfill(4)+'_R='+str(r).zfill(4)+'.npy')
                        dd[t].append(1.0*datn[l]/datr[l])

        plt.clf()
        plt.subplot(121)
        plt.suptitle(label)
        img2net_help.boxplot(dn,'black',treatments,0)
        plt.ylabel('absolute')
        plt.xlabel('treatments')
        plt.subplot(122)
        plt.plot([0,T+1],[1,1],lw=2,color='gray',ls='--')
        img2net_help.boxplot(dd,'black',treatments,1)
        plt.ylabel('relative')
        plt.xlabel('treatments')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(dir_output+'plot_dist/plot_dist_p='+str(l).zfill(4)+'_'+label+'.svg')
        #plt.show()

        ################################################################# plot times series

        plt.clf()
        T2=np.where(T>2,T,2).min()
        use=0.40/T
        gap=0.60/(T2-1.0)
        for t in range(T):
            E2=np.where(E[t]>2,E[t],2).min()
            for e in range(E[t]):
                data=[[] for i in range(I[t][e])]
                for i in range(I[t][e]):
                    data[i]=np.load(dir_output+'data_datn/data_datn_T='+str(t).zfill(4)+'_E='+str(e).zfill(4)+'_I='+str(i).zfill(4)+'.npy')[l]
                sec=t*use+t*gap+e*use/(E2-1.0)
                plt.title(label)
                plt.plot(data,lw=2,color=plt.cm.jet(sec),label=treatments[t]+','+experiments[t][e])
                plt.xlabel('frame')
                plt.ylabel('absolute')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(dir_output+'plot_time/plot_time_p='+str(l).zfill(4)+'_'+label+'.svg')

        s+=1.0
        self.builder.get_object('progressbar2').set_fraction(s/S)
        while gtk.events_pending():
            gtk.main_iteration()

    ################################################################# plot example grids

    s+=1.0
    self.builder.get_object('progressbar2').set_text('saving grids')
    self.builder.get_object('progressbar2').set_fraction(s/S)
    while gtk.events_pending():
        gtk.main_iteration()

    for t in range(T):
        for e in range(E[t]):
            dir_te=dir_input+treatments[t]+'/'+experiments[t][e]+'/'
            ims=sp.ndimage.imread(dir_te+images[t][e][lz-1])
            ly,lx=np.shape(ims)[:2]
            name=dir_output+'data_grph/data_grph_T='+str(t).zfill(4)+'_E='+str(e).zfill(4)+'_I='+str(0).zfill(4)+'.npy'
            gn=nx.from_edgelist(np.load(name))
            N=gn.number_of_nodes()
            subgraph=nx.subgraph(gn,range(N-N/lz,N))
            gc=nx.convert_node_labels_to_integers(subgraph,ordering='sorted')
            posi=np.load(dir_output+'data_posi/data_posi_T='+str(t).zfill(4)+'_E='+str(e).zfill(4)+'.npy').flatten()[0]
            pos2D,pos3D=img2net_help.grid_pos2D(N,lz,posi)
            en=np.array([d['capa'] for u,v,d in gn.edges(data=True)])
            en=en/en.max()
            ec=np.array([d['capa'] for u,v,d in gc.edges(data=True)])
            ec=ec/en.max()
            plt.clf()
            plt.subplot(121)
            plt.imshow(ims,cmap='gray',origin='lower',extent=[0,lx,0,ly])
            nx.draw_networkx_edges(gc,pos2D,width=2,edge_color=ec)
            plt.axis('off')
            fig=plt.subplot(122,projection='3d',axisbg='white')
            fig.view_init(elev=30,azim=20)
            L=gn.number_of_edges()
            edges=list(gn.edges())
            for l in range(L):
                [u,v]=edges[l][0:2]
                if(posi[u][2]!=posi[v][2]):
                    alp=0.4
                else:
                    alp=1.0
                plt.plot([posi[u][1],posi[v][1]],[posi[u][0],posi[v][0]],[posi[u][2],posi[v][2]],color=plt.cm.jet(en[l]),alpha=alp,lw=2)
            plt.axis('off')
            plt.savefig(dir_output+'plot_grid/plot_grid_t='+str(t).zfill(2)+'_e='+str(e).zfill(2)+'.svg')

            s+=1.0
            self.builder.get_object('progressbar2').set_fraction(s/S)
            while gtk.events_pending():
                gtk.main_iteration()

    print('s',s,S,'c',c,C)

    self.builder.get_object('progressbar1').set_text('  ')
    self.builder.get_object('progressbar2').set_text('  ')
    self.builder.get_object('progressbar1').set_fraction(0)
    self.builder.get_object('progressbar2').set_fraction(0)
    while gtk.events_pending():
        gtk.main_iteration()

    return 0
