#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:46:32 2019

@author: antonio
"""
#documento para el 1-knn http://www.sc.ehu.es/ccwbayes/docencia/mmcc/docs/t9knn.pdf
from math import sqrt
from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold

SEMILLA = 42
#carga los datos y los devuelve normalizados
def cargarDatos(ubicacion, typedat):

	_data_arff = arff.loadarff(ubicacion)


	_data = pd.DataFrame(_data_arff[0])
	_target = _data["class"]
	_target = _target.to_frame()
	_target=_target.astype(typedat)
	_data = _data.drop("class", axis=1)

	#Normalizo los datos de entrada
	_min_max_scaler = preprocessing.MinMaxScaler()
	_np_scaled = _min_max_scaler.fit_transform(_data)
	_Norm = pd.DataFrame(_np_scaled)
	return _Norm.values, _target.values

def abs(num):
	if num < 0:
		num = -num
	return num

def distancia(_X, pos1, pos2):
	distancia = 0
	for _x, _column in enumerate(_X[pos1, : ]):
		distancia += (_column - _X[pos2, _x])*(_column - _X[pos2, _x])
	return sqrt(distancia)

def clasificador(_X,_y, epos):
	cmin = _y[0]
	dmin = distancia(_X,_y, 0,epos)
	for _x in range(1, _X.size):
		d = distancia(_X,_y, _x,epos)
		if d < dmin:
			cmin = _y[_x]
			dmin = d
	return cmin

def RELIEF(_data, _target):
	_W = np.zeros(_data[0, : ].shape)
	_distanceMatriz = euclidean_distances(_data, _data)
	for _x, _fila in enumerate(_data):
		#recorro todos buscando enemigos
		_mejorDistancia = float('Inf')
		enemigoMasCercano = -1
		for _y1, _fila1 in enumerate(_data):
			if(_target[_y1] != _target[_x]):
				if _distanceMatriz[_x,_y1] < _mejorDistancia:
					_mejorDistancia = _distanceMatriz[_x,_y1]
					enemigoMasCercano = _fila1

		#recorro todos buscando amigos
		_mejorDistancia = float('Inf')
		_amigoMasCercano = -1
		for _y2, _fila2 in enumerate(_data):
			if(_target[_y2] == _target[_x]):
				if _distanceMatriz[_x,_y2] < _mejorDistancia:
					_mejorDistancia = _distanceMatriz[_x,_y2]
					_amigoMasCercano = _fila2

		_W = _W + np.absolute(_fila-enemigoMasCercano) - np.absolute(_fila - _amigoMasCercano)

	maxValW=max(_W)
	for dato in range(_W.__len__()):
		if(_W[dato]<0):
			_W[dato]=0
		else:
			_W[dato]=_W[dato]/maxValW

	return _W

TEXTURE = "ConjuntosDatos/texture.arff"
COLPOSCOPY = "ConjuntosDatos/colposcopy.arff"
IONOSPHERE = "ConjuntosDatos/ionosphere.arff"

#data, target = cargarDatos(COLPOSCOPY,'int')
data, target = cargarDatos(IONOSPHERE,'str')

#relief
w  = RELIEF(data, target)


"""

skf = StratifiedKFold(n_splits=5, shuffle = True,  random_state = SEMILLA)
skf.get_n_splits(data, target)

print(skf)
for train_index, test_index in skf.split(data, target):
	print("TRAIN:", train_index, "TEST:", test_index)
	if cont == 0:
		X_train, X_test = data[train_index], data[test_index]
		y_train, y_test = target[train_index], target[test_index]
"""