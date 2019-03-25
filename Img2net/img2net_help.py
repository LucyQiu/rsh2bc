##################################################################### imports

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import scipy as sp
import scipy.ndimage
import scipy.stats
import warnings
##################################################################### functions

def help_periodicdistance(lx,x1,x2,pbc):
    if(pbc==0):
        dx=np.abs(x1-x2)
    else:
        dx=np.array([np.abs(x1-x2),np.abs(x1-x2+lx),np.abs(x1-x2-lx)]).min()
    return dx

def help_periodiclist(lx,x1,pbc):
    if(pbc==0):
        dx=np.abs(np.subtract(range(lx),x1))
    else:
        dx=np.abs(np.minimum(np.minimum(np.abs(np.subtract(range(lx),x1)),np.abs(np.subtract(range(lx),lx+x1))),np.abs(np.subtract(range(lx),-lx+x1))))
    return dx

def help_edgekernel(lx,ly,vax,vay,x1,y1,x2,y2,pbcx,pbcy):
    dx1=help_periodiclist(lx,x1,pbcx)
    dx2=help_periodiclist(lx,x2,pbcx)
    dy1=help_periodiclist(ly,y1,pbcy)
    dy2=help_periodiclist(ly,y2,pbcy)
    ex1=np.ones((ly,1))*dx1**2/(2.0*vay)
    ey1=np.transpose(np.ones((lx,1))*dy1**2/(2.0*vax))
    ex2=np.ones((ly,1))*dx2**2/(2.0*vay)
    ey2=np.transpose(np.ones((lx,1))*dy2**2/(2.0*vax))
    ek=np.exp(-np.sqrt(ex1+ey1)-np.sqrt(ex2+ey2))
    return np.divide(ek,np.nansum(ek))

def help_angle(vec):
    if(vec[0]<0):
        vec=-vec
    angle=180.0/np.pi*np.arccos(vec.dot([0,1])/np.sqrt(vec.dot(vec)))
    return angle

def grid_pos2D(N,nnz,pos):
    pos2D={}
    pos3D={}
    for n in range(N/nnz):
        pos2D[n]=(pos[n][0],pos[n][1])
    for n in range(N):
        pos3D[n]=(pos[n][0],pos[n][1])
    return pos2D,pos3D

def grid_grid(crd,lx,ly,lz,dx,dy,dz,pbx,pby,pbz,vax,vay,compute_convs,dir_posi,dir_conv):

    pos={}
    edges=[]
    h=np.sqrt(3.0)/2.0
    dh=dy*h
    nnx=int(lx/dx)
    nny=int(ly/dy)
    nnh=int(ly/dh)
    nnz=int(lz)
    DX=0.5*(lx-(nnx-1)*dx)
    DY=0.5*(ly-(nny-1)*dy)
    DH=0.5*(ly-(nnh-1)*dh)

    if(crd=='rectangular'):
        N=nnx*nny*nnz
        n=0
        for z in range(nnz):
            for y in range(nny):
                for x in range(nnx):
                    pos[n]=(x*dx+DX,y*dy+DY,z*dz)
                    n+=1
        dh=dy

    elif(crd=='triangular'):
        nnx=nnx-1+np.mod(nnx,2)
        nnh=nnh-np.mod(nnh,2)
        N=nnx*nnh*nnz
        n=0
        for z in range(nnz):
            for h in range(nnh):
                for x in range(nnx):
                    pos[n]=((x+0.5*np.mod(h,2))*dx+0.5*DX,(h+0.5)*dh+DH,z*dz)
                    n+=1
        nny=nnh

    elif(crd=='hexagonal'):
        nnx=3*int(nnx/3)
        nnh=nnh-np.mod(nnh,2)
        N=nnx*nnh*nnz
        n=0
        for z in range(nnz):
            for h in range(nnh):
                for x in range(nnx):
                    if(np.mod(h,2)==0 and np.mod(x+2,3)!=0):
                        pos[n]=((x+0.5+0.5*np.mod(h,2))*dx+0.5*DX,(h+0.5)*dh+DH,z*dz)
                        n+=1
                    if(np.mod(h,2)==1 and np.mod(x+1,3)!=0):
                        pos[n]=((x+0.5+0.5*np.mod(h,2))*dx+0.5*DX,(h+0.5)*dh+DH,z*dz)
                        n+=1
        N=n
        nny=nnh

    for n in range(N):
        for m in range(n):
            Dx=help_periodicdistance(nnx*dx,pos[n][0],pos[m][0],pbx)
            Dy=help_periodicdistance(nny*dh,pos[n][1],pos[m][1],pby)
            Dz=help_periodicdistance(nnz*dz,pos[n][2],pos[m][2],pbz)
            if((Dx/dx)**2+(Dy/dy)**2+(Dz/dz)**2<1.1):
                edges.append((n,m))

    E=len(edges)
    pos2D,pos3D=grid_pos2D(N,nnz,pos)

    convs=[]
    if(compute_convs):
        np.save(dir_posi,pos)
        for e in range(E):
            n=edges[e][0]
            m=edges[e][1]
            x1=pos3D[n][0]
            y1=pos3D[n][1]
            x2=pos3D[m][0]
            y2=pos3D[m][1]
            conv=help_edgekernel(lx,ly,vax,vay,x1,y1,x2,y2,pbx,pby)
            np.save(dir_conv+'_L='+str(e).zfill(4),conv)
            if(E<1000):
                convs.append(conv)

    return N,nnx,nny,nnz,pos,E,edges,convs



def grid_graph(im,E,edges,dir_pos,dir_convs,dz):
    capas=[]
    pos=np.load(dir_pos+'.npy').flatten()[0]
    for e in range(E):
        conv=np.load(dir_convs+'_L='+str(e).zfill(4)+'.npy')
        n=edges[e][0]
        m=edges[e][1]
        z0=int(pos[n][2]/dz)
        z1=int(pos[m][2]/dz)
        if(z0!=z1):
            capas.append(np.nansum(np.multiply(0.5*(im[z0]+im[z1]),conv)))
        else:
            capas.append(np.nansum(np.multiply(im[z0],conv)))
    capas=np.divide(capas,np.nansum(capas))

    graph=nx.Graph()
    for e in range(E):
        n=edges[e][0]
        m=edges[e][1]
        graph.add_edge(n,m,capa=capas[e],lgth=1.0/capas[e])
    return graph

def grid_all(inp):
    t,e,i,directory,files,L,edges,dir_pos,dir_convs,dz=inp
    ims=[]
    for z in range(len(files)):
        im=1.0*sp.ndimage.imread(directory+files[z])
        if(len(np.shape(im))>2):
            im=im[:,:,0]
        ims.append(im)
    graph=grid_graph(ims,L,edges,dir_pos,dir_convs,dz)

    return t,e,i,graph

def graph_null(graphn,nnx,nny,numo):
        graphg=graphn.copy()

        if(numo=='edges'):
            E=graphg.number_of_edges()
            idx=range(E)
            random.shuffle(idx)
            edges = list(graphn.edges(data=True))
            for e,(u,v,d) in enumerate(graphg.edges(data=True)):
                d['capa']=edges[idx[e]][2]['capa']
                d['lgth']=edges[idx[e]][2]['lgth']

        elif(numo=='lines'):
            E=graphg.number_of_edges()
            idx=range(nnx)
            idy=range(nny)
            random.shuffle(idx)
            random.shuffle(idy)
            links=graphg.edges(data=True)
            for e in range(E):
                [n0,n1]=links[e][0:2]
                [p0x,p0y]=[np.mod(n0,nnx),n0/nnx]
                [p1x,p1y]=[np.mod(n1,nnx),n1/nnx]
                if(p0x==p1x):
                    graphg.edges(data=True)[e][2]['capa']=graphn[idx[p0x]+p0y*nnx][idx[p1x]+p1y*nnx]['capa']
                    graphg.edges(data=True)[e][2]['lgth']=graphn[idx[p0x]+p0y*nnx][idx[p1x]+p1*nnx]['lgth']
                if(p0y==p1y):
                    graphg.edges(data=True)[e][2]['capa']=graphn[p0x+idy[p0y]*nnx][p1x+idy[p1y]*nnx]['capa']
                    graphg.edges(data=True)[e][2]['lgth']=graphn[p0x+idy[p0y]*nnx][p1x+idy[p1y]*nnx]['lgth']

        elif(numo=='blocks'):
            Bx,By=3,4
            graphg=graphn.copy()
            E=graphg.number_of_edges()
            N=graphg.number_of_nodes()
            bx=nnx/Bx
            by=nny/By
            B=Bx*By
            idb=range(B)
            random.shuffle(idb)
            links=graphg.edges(data=True)
            left=set(range(E))
            for e in range(E):
                [n0,n1]=links[e][0:2]
                [p0x,p0y]=[np.mod(n0,nnx),n0/nnx]
                [p1x,p1y]=[np.mod(n1,nnx),n1/nnx]
                [c0x,c0y]=[p0x/bx,p0y/by]
                [c1x,c1y]=[p1x/bx,p1y/by]
                [c0,c1]=[c0x+c0y*Bx,c1x+c1y*Bx]
                if(c0==c1 and c0<B):
                    left=left.difference([e])
                    [cnewx,cnewy]=[np.mod(idb[c0],Bx),idb[c0]/Bx]
                    [d0x,d0y]=[p0x+(cnewx-c0x)*bx,p0y+(cnewy-c0y)*by]
                    [d1x,d1y]=[p1x+(cnewx-c1x)*bx,p1y+(cnewy-c1y)*by]
                    [m0,m1]=[d0x+d0y*nnx,d1x+d1y*nnx]
                    graphg.edges(data=True)[e][2]['capa']=graphn[m0][m1]['capa']
                    graphg.edges(data=True)[e][2]['lgth']=graphn[m0][m1]['lgth']
            left=list(left)
            ew=[graphg.edges(data=True)[i][2]['capa'] for i in left]
            el=[graphg.edges(data=True)[i][2]['lgth'] for i in left]
            random.shuffle(left)
            for e,ee in enumerate(left):
                graphg.edges(data=True)[ee][2]['capa']=ew[e]
                graphg.edges(data=True)[ee][2]['lgth']=el[e]

        return graphg

def graph_all(inp):
    [t,e,i,r,G,pos]=inp
    data=[]
    label=[]
    N=G.number_of_nodes()

    ################################################# degree
    deg=[val for node, val in G.degree(weight='capa')]

    label.append('mean[degree]')
    data.append(np.nanmean(deg))
    label.append('sd[degree]')
    data.append(np.nanstd(deg))
    label.append('skewness[degree]')
    data.append(sp.stats.skew(deg))

    ################################################# structure

    label.append('clustering coefficient')
    data.append(nx.average_clustering(G,weight='capa'))
    label.append('assortativity')
    data.append(nx.degree_pearson_correlation_coefficient(G,weight='capa'))

    ################################################# distances
    dists=nx.all_pairs_dijkstra_path_length(G,weight='lgth')
    dist=[[valv for keyv, valv in valu.items()] for keyu, valu in dists]
    ecce=np.nanmax(np.array(dist), axis=0)

    label.append('mean[distance]')
    data.append(np.nanmean(dist))
    label.append('sd[distance]')
    data.append(np.nanstd(dist))
    label.append('skewness[distance]')
    data.append(sp.stats.skew(np.reshape(dist,-1)))
    label.append('radius')
    data.append(ecce.min())
    label.append('diameter')
    data.append(ecce.max())

    ################################################# eigenvalues
    try:
        spec=np.sort(nx.laplacian_spectrum(G,weight='capa'))
    except:
        warnings.warn('Computation of Laplacian spectrum failed, effective resistance and algebraic connectivity are not reliable.')
        spec=np.ones(N)

    label.append('effective resistance')
    data.append(1.0/np.nansum(np.divide(1.0*N,spec[1:N-1])))
    label.append('algebraic connectivity')
    data.append(spec[1])

    ################################################# betweenness
    flow=nx.edge_current_flow_betweenness_centrality(G,weight='capa',normalized=1).values()

    label.append('mean[betweenness]')
    data.append(np.nanmean(flow))
    label.append('sd[betweenness]')
    data.append(np.nanstd(flow))
    label.append('skewness[betweenness]')
    data.append(sp.stats.skew(flow))

    ################################################# angles
    angle_angle=[]
    angle_weight=[]
    for u,v,d in G.edges(data=True):
        angle_angle.append(help_angle(np.subtract(pos[u][0:2],pos[v][0:2])))
        angle_weight.append(d['capa'])
    angle_angle=np.mod(angle_angle,180)

    label.append('angle 000')
    data.append(np.nansum(np.where(angle_angle==0,angle_weight,0))/np.nansum(np.where(angle_angle==0,1,0)))
    label.append('angle 045')
    data.append(np.nansum(np.where(angle_angle==45,angle_weight,0))/np.nansum(np.where(angle_angle==45,1,0)))
    label.append('angle 060')
    data.append(np.nansum(np.where(angle_angle==60,angle_weight,0))/np.nansum(np.where(angle_angle==60,1,0)))
    label.append('angle 090')
    data.append(np.nansum(np.where(angle_angle==90,angle_weight,0))/np.nansum(np.where(angle_angle==90,1,0)))
    label.append('angle 120')
    data.append(np.nansum(np.where(angle_angle==120,angle_weight,0))/np.nansum(np.where(angle_angle==120,1,0)))
    label.append('angle 135')
    data.append(np.nansum(np.where(angle_angle==135,angle_weight,0))/np.nansum(np.where(angle_angle==135,1,0)))
    label.append('angle ratio 00-90')
    data.append(data[-6]/data[-3])

    return t,e,i,r,data,label


def boxplot(data,color,labels,ttest):
    L=len(labels)
    bp=plt.boxplot(data,sym='',notch=0,widths=0.5)
    [plt.setp(bp[k],color=color,ls='-',alpha=1.0,lw=2.0) for k in bp.keys()]
    lims=np.array([bp['whiskers'][l].get_data()[1] for l in range(2*L)])
    plt.xticks(range(1,1+L),labels)
    limi,lima=plt.ylim()
    limd=lima-limi
    plt.ylim([limi,lima+0.1*limd])
    if(ttest==1):
        for i in range(len(labels)):
            pval=sp.stats.ttest_1samp(data[i],1)[1]
            plt.text(i+1,lima+0.03*limd,'p=%.2f'%pval,ha='center')
    return None

