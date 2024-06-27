#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:53:35 2018

@author: iG
"""

import pandas as pd
import numpy as np
import tensorflow.keras.models as models
import matplotlib.pyplot as plt
import matplotlib.colors
import neighdist
import itertools

plt.rcParams['figure.figsize']=(12,9)
plt.rcParams['font.size'] = 16.0
plt.rcParams['font.family'] = 'sans-serif'

cmap = plt.cm.get_cmap('RdYlBu')
norm = matplotlib.colors.BoundaryNorm(np.arange(0,1.1,0.1), cmap.N)

"""An auxiliary file used to compute the features needed by the ANN"""
datos = pd.read_csv('datosrahm.csv')

maindict = {}
for row in range(datos.shape[0]):
    maindict[datos['Symbol'][row]] = \
    datos.iloc[row,:][['elecPau','atradrahm']].values

"""The Artificial Neural Network is loaded as well as the parameters to do the feature standarization"""
model_dict = np.load('dictionary_upto_4Wyckoffsites.npy').item()
model = models.load_model('patolli_upto_4Wyckoffsites.h5')

wyckcub = {0:{'A' : np.asarray([[0.0,0.0,0.0]])},
          1:{'B' : np.asarray([[0.5,0.5,0.5]])},
          2:{'X' : np.asarray([[0.5,0.5,0.0],[0.5,0.0,0.5],[0.0,0.5,0.5]])}}

def compute_quotients(X = np.zeros((1,1,2))):
    """
    Returns the atomic radii pair quotients and the atomic radii 
    pair sum - quotients as a numpy array. Thjs is the first part of
    all the features used to train the ANNs
    Parameters:
        X: A numpy array, which is created with the function raw_features_extractor
    Returns:
        X: A numpy array of dimension [samples,1,features]
    """
    
    rad = X[:,:,1]

    X = np.reshape(X,(X.shape[0],1,X.shape[1]*X.shape[2]))

    drad = np.asarray([[item[0]/item[1] if item[1] != 0 else 0 for item in list(itertools.combinations(rad[sample],2))] \
                        for sample in range(X.shape[0])])

    dradsum = np.asarray([[item[0]/item[1] if item[1] != 0 else 0 for item in itertools.combinations([ \
                       item[0]+item[1] for item in list(itertools.combinations(rad[sample],2))], 2)] \
                       for sample in range(drad.shape[0])])
    
    drad = np.reshape(drad,(drad.shape[0],1,drad.shape[-1]))
    drads = np.reshape(dradsum,(dradsum.shape[0],1,dradsum.shape[-1]))

    X = np.concatenate((drad,drads), axis=2)
    print('Geometric and packing factors computed')
        
    return X
    
def scan_nnoutputs(elements = [['Na'],['V'], ['O']],
                   compositions = [[1.0], [1.0],[1.0]], 
                   maxdev = 0.2, stepsize = 0.01, 
                   name = ''):
    """
    This function assesses the probability to crystallize as a perovskite structure
    among different lattice parameter. The perovskite structures are considered in the
    aristotype form (space group No. 221). 
    Besides, csv- and png- files are saved containing the probabilities against the
    explored lattice parameters.
    Parameters: 
    	elements: a list of list, which specifies the atoms located in the cuboctahedral,
    			octahedral and vertex sites.
    	compositions: a list of list, which indicates the ocupation fraction of each atom 
    			in the sites.
    	maxdev: float, maximum deviation from the calculated lattice parameter. The lattice
    			parameter is proposed to be the sum of the atomic radii of the atoms in the
    			octahedral and vertex sites. Default, 0.2.
    	stepsize: float, stepsize used to scan the perovskite probability among the range
    			defined by the maxdev. Default,  0.01.
    	name: string, the name for the saved files. If it is not given, the name will be
    			the formula of the simulated compound.
    Returns:
    	y: Numpy array, containing the probabilies to crystallize as a perovskite structure.
    	scanned_latpar: Numpy array, containing the lattice parameter used to perform the simulation.
    """   
    
    scann = np.arange(1 - maxdev, 
                      1 + maxdev + stepsize, 
                      stepsize)
    
    multiplicities = [[1],[1],[3]]
    sidx = [np.asarray(i)*np.asarray(j) for i,j in zip(compositions, multiplicities)]
    sidx = [k for h in sidx for k in h]
    
    elements2 = [k for j in elements for k in j]
    formula = [i + '$_{' + "%.2f" % j + '}$' \
               for i,j in zip(elements2, sidx)]
    formula = ''.join([i for i in formula])
    
    primfeat = [np.dot(np.asarray(subind), np.asarray([maindict.get(item,None) \
              for item in site])) for site, subind in zip (elements, compositions)]
    
    primfeat = np.asarray(primfeat)
    primfeat = np.concatenate((np.zeros((1,primfeat.shape[1])),primfeat), axis = 0)
    
    latpar = primfeat[2][1] + primfeat[3][1]

    multiplicities = np.asarray(multiplicities)

    primfeat = primfeat.reshape((1, primfeat.shape[0],primfeat.shape[1]))
    x = compute_quotients(X=primfeat)
    
    scann = scann**3

    scanned_latpar = latpar*(scann**(1/3))
 
    dist = 25
    
    radii = primfeat[:,:,1]
    
    elec = primfeat[:,:,0]
    elec = elec.reshape((4,1))
    
    delec = np.repeat(elec[:,np.newaxis],4,axis=2) - \
            np.repeat(elec[:,np.newaxis],4,axis=2).T
    delec = delec.reshape((delec.shape[0],delec.shape[2]))
    
    fr = np.zeros((len(scann),4,4-1))
    for item in range(len(scanned_latpar)):
        
        p, z, n, m = neighdist.positions(pos = wyckcub, 
                                         angles = [90,90,90], 
                                         abc = [scanned_latpar[item],]*3,
                                         dist = dist)
        r = neighdist.rij(mult=m,p=p,zero=z,
                          dist=dist, radii = np.ravel(radii))
        
        temp = np.multiply(r,delec)
        temp = temp[~np.eye(temp.shape[0],
                      dtype=bool)].reshape(temp.shape[0],-1)
        fr[item] = temp
            
    fr = fr.reshape((fr.shape[0],1,fr.shape[1]*fr.shape[2]))
    
    x = np.asarray((x,)*len(scann))
    x = x.reshape((x.shape[0],1,x.shape[3]))
    x = np.concatenate((x,fr), axis=2)
    
    X = (x - model_dict['mean'])/model_dict['std']

    y = model.predict(X.reshape((X.shape[0],X.shape[2])))
    
    if not name:
        name = formula.replace('$_{','').replace('}$','') + '_ann_prob'
    
    plt.figure()
    plt.title(r'Perovskite probability for ' + formula)
    plt.scatter(scanned_latpar, np.round(100*y[:,0],2), 
                marker='o', s=75,  color = '#a60b20')
                
    plt.xlabel('lattice parameter')
    plt.ylabel('ANN output probability')
    plt.savefig(name + '.png')
    
    with open(name + '.csv', 'w') as f:
        
        f.write('lattice parameter (angstrom),perovskite probability (%)' + '\n' )
        
        for item in range(scanned_latpar.shape[0]):
        
            probability = 100*y[item,0]
                        
            f.write("%.4f" % scanned_latpar[item] + ',')
            f.write("%.2f" % probability)
            f.write('\n')
        f.close()
        
    print('file saved for the compound', formula.replace('$_{','').replace('}$',''),
          'with the name', name)
    return y, scanned_latpar


def ctrl_dictionary(archivo='compounds2simulate'):
    """
    Parameters: A txt - file which has the compounds to simulate with the ANN.
    Return: A dictionary used afterwards with the function scan_nnoutputs.
    """
    
    f=list(filter(None,open(str(archivo)+'.txt','r').read().split('\n')))

    sg_ikeys=[f.index(sg) for sg in f if 'COMPOUND' in sg]+[len(f)]
    
    diccio={}
    for item in range(len(sg_ikeys)-1):
        text = f[sg_ikeys[item]:sg_ikeys[item+1]]
        key = [entry.split(':')[0] for entry in text]
        value = [entry.split(':')[1] for entry in text]
        diccio[item] = {k:v for k,v in zip(key,value)}
    
    dicciovar = {}
    for key in range(len(diccio.keys())):
        dicciovar[key] = dict()
        dicciovar[key]['name'] = diccio[key]['COMPOUND'].lstrip()
        dicciovar[key]['elements'] = list()
        dicciovar[key]['compositions'] = list()
        dicciovar[key]['maxdev'] = float(diccio[key]['maxdev'])
        dicciovar[key]['stepsize'] = float(diccio[key]['stepsize'])
        
        dicciovar[key]['elements'] += [[i.lstrip() for i in diccio[key]['cuboctahedron_atom'].split(',')]]
        dicciovar[key]['elements'] += [[i.lstrip() for i in diccio[key]['octahedron_atom'].split(',')]]
        dicciovar[key]['elements'] += [[i.lstrip() for i in diccio[key]['vertex_atom'].split(',')]]
        
        dicciovar[key]['compositions'] += [[float(i) for i in diccio[key]['cuboctahedron_frac'].split(',')]]
        dicciovar[key]['compositions'] += [[float(i) for i in diccio[key]['octahedron_frac'].split(',')]]
        dicciovar[key]['compositions'] += [[float(i) for i in diccio[key]['vertex_frac'].split(',')]]
        
    
    return dicciovar


archivo = input('Please provide the name of the text file containing ' +  
                'the compounds to simulate [Default compounds2simulate]:' + 
                '\n')
if not archivo:
    archivo = 'compounds2simulate'
    
filediccio = ctrl_dictionary(archivo=archivo)

for key in filediccio:
    scan_nnoutputs(elements = filediccio[key]['elements'], 
                   compositions = filediccio[key]['compositions'], 
                   maxdev = filediccio[key]['maxdev'], 
                   stepsize = filediccio[key]['stepsize'], 
                   name = filediccio[key]['name'])