#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:46:32 2019

@author: antonio
"""
#documento para el 1-knn http://www.sc.ehu.es/ccwbayes/docencia/mmcc/docs/t9knn.pdf
from math import sqrt

import time
from beautifultable import BeautifulTable
import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KDTree

#para obtener el tiempo en milisegundos
current_milli_time = lambda: int(round(time.time() * 1000))

SEMILLA = 1
#carga los datos y los devuelve normalizados
def cargarDatos(ubicacion, typedat):

	#cargo los fichero
	_data_arff = arff.loadarff(ubicacion)
	_data = pd.DataFrame(_data_arff[0])
	_target = _data["class"]
	_target = _target.to_frame()
	#cambio el tipo de de dato las clases
	_target=_target.astype(typedat)
	#elimino la columna de clase de los datos, para tenerlos separados
	_data = _data.drop("class", axis=1)

	#Normalizo los datos de entrada
	_min_max_scaler = preprocessing.MinMaxScaler()
	_np_scaled = _min_max_scaler.fit_transform(_data)
	_Norm = pd.DataFrame(_np_scaled)
	return _Norm.values, _target.values




#algoritmo relief que devuelve el vector de pesos para el modelo
def RELIEF(_data, _target):
	_W = np.zeros(_data[0, : ].shape)
	#genero una matriz de distancia para tardar menos en la ejecución
	_distanceMatriz = euclidean_distances(_data, _data)
	#recorro todos los datos
	for _x, _fila in enumerate(_data):
		#inicializo variables de distancias y su valor correspondiente para amigo y enemigo
		_mejorenemigo = float('Inf')
		_mejorAmigo = float('Inf')
		_enemigoMasCercano = -1
		_amigoMasCercano = -1

		#recorro todos los datos buscando su amgigo y enemigo mas cercano
		for _y1, _fila1 in enumerate(_data):
			if(_target[_y1] != _target[_x]):
				if _distanceMatriz[_x,_y1] < _mejorenemigo:
					_mejorenemigo = _distanceMatriz[_x,_y1]
					_enemigoMasCercano = _fila1
			#tengo en cuenta que no se compare a el mismo
			if(_target[_y1] == _target[_x]) and _y1 != _x:
				if _distanceMatriz[_x,_y1] < _mejorAmigo:
					_mejorAmigo = _distanceMatriz[_x,_y1]
					_amigoMasCercano = _fila1
		#modifico el valor de pesos en funcion de la formola del relief
		_W = _W + np.absolute(_fila-_enemigoMasCercano) - np.absolute(_fila - _amigoMasCercano)

	#normalizo los datos eliminando incluso los que tengan un peso menor que 0.2
	maxValW=max(_W)
	for dato in range(_W.__len__()):
		if(_W[dato]<0.2):
			_W[dato]=0
		else:
			_W[dato]=_W[dato]/maxValW

	return _W

def ejecucionRelief(_data, _target):
	_inicio = current_milli_time()
	_time_actual = _inicio	#Realizo las 5 particiones con mezcla y semilla
	skf = StratifiedKFold(n_splits=5, shuffle = True,  random_state = SEMILLA)
	skf.get_n_splits(_data, _target)
	#recorro cada una de las
	sumatoriaClas = 0
	sumatoriaRed = 0
	sumatoriaAgregada = 0
	_cont = 1
	tabla = BeautifulTable()
	tabla.column_headers = ["Particion","%cls","%redu","%agr", "tiempo"]
	for _train_index, _test_index in skf.split(_data, _target):
		#obtengo las particiones de test y train
		X_train, _X_test = _data[_train_index], _data[_test_index]
		y_train, _y_test = _target[_train_index], _target[_test_index]
		#obtengo los pèsos para esta particion
		_w  = RELIEF(X_train, y_train)
		#obtengo las tasas  de evaular la solución
		tasa_clas, tasa_red, tasa_agr = evaluate(_w, _X_test, _y_test)
		sumatoriaClas += tasa_clas
		sumatoriaRed += tasa_red
		sumatoriaAgregada += tasa_agr
		time_anterior= _time_actual
		_time_actual =current_milli_time()
		tabla.append_row([_cont, tasa_clas, tasa_red,tasa_agr,( (_time_actual - time_anterior) / 1000.0)])

		_cont += 1

	_final = current_milli_time()
	#calculo el tiempo de ejecucion
	diferencia =(_final- _inicio) / 1000.0
	#muestro los estadisticos
	tabla.append_row(["Media",sumatoriaClas/5,sumatoriaRed/5,sumatoriaAgregada/5,(diferencia/5)])
	print(tabla)

def distancia(X, pos1, _fila):
	return sqrt(np.sum(np.power(X[pos1]-_fila, 2)))
def clasificador(X,_y, _fila):
	cmin = _y[0]
	dmin = distancia(X,0, _fila)
	for _x in range(1, X.shape[0]-1):
		d = distancia(X,_x,_fila)
		if d < dmin:
			cmin = _y[_x]
			dmin = d
	return cmin

def knn(datos, etiquetas, datos_test, etiquetasTest):
	aciertos = 0

	for i, idx in enumerate(etiquetasTest):

		test = datos_test[i]
		test = test.reshape(1, -1)

		if clasificador(datos,etiquetas,test) == idx:
			aciertos += 1
	acury =  aciertos / len(datos_test)
	return 100*acury
def ejecutarKNN(_data, _target):
	_inicio = current_milli_time()
	_time_actual = _inicio
	#Realizo las 5 particiones con mezcla y semilla
	skf = StratifiedKFold(n_splits=5, shuffle = True,  random_state = SEMILLA)
	skf.get_n_splits(_data, _target)
	#recorro cada una de las
	sumatoriaClas = 0
	_cont = 1
	tabla = BeautifulTable()
	tabla.column_headers = ["Particion","%cls", "tiempo"]
	for _train_index, _test_index in skf.split(_data, _target):
		#obtengo las particiones de test y train
		X_train, _X_test = _data[_train_index], _data[_test_index]
		y_train, _y_test = _target[_train_index], _target[_test_index]
		#obtengo los pèsos para esta particion

		#obtengo las tasas  de evaular la solución
		tasa_clas = knn(X_train, y_train, _X_test,_y_test)
		time_anterior= _time_actual
		_time_actual =current_milli_time()
		sumatoriaClas += tasa_clas

		tabla.append_row([_cont, tasa_clas,( (_time_actual - time_anterior) / 1000.0)])

		_cont += 1

	_final = current_milli_time()
	#calculo el tiempo de ejecucion
	diferencia =(_final- _inicio) / 1000.0
	#muestro los estadisticos
	tabla.append_row(["Media", tasa_clas/5,diferencia/5])
	print(tabla)

def evaluate(weights, X, y):
    X_transformed = (X * weights)[:, weights > 0.2]
    kdtree = KDTree(X_transformed)
    neighbours = kdtree.query(X_transformed, k=2)[1][:, 1]
    accuracy = np.mean(y[neighbours] == y)
    reduction = np.mean(weights < 0.2)
    return 100*accuracy,reduction*100,100*(accuracy + reduction) / 2

def BL(_train, _y_train):


	#vector de pesos inicial generado con numeros aleatorios y su valor actual
	W = np.random.random(_train[0].size)
	#prueba = evalu(W,_train,_y_train)
	acu, red, mejor_tasa_agr = evaluate(W,_train,_y_train)
	_MAX = 15000

	_cantCaractarteristicas = W.size
	_cantEjecuciones = 0
	_cantExplorados = 0

	while(_cantEjecuciones < _MAX and _cantExplorados < (20*_cantCaractarteristicas)):
		for y in range(W.size):
			#he explorado uno mas y la cantidad de ejecuciones es una mas
			_cantExplorados += 1
			_cantEjecuciones += 1
			#creo una copia del _W anterior y lo muto
			W_anterior =  W[y]
			Z =  np.random.normal(0.0, 0.3)
			W[y] += Z

			#normalizo los datos segun pone en las transparencias
			if W[y]>1:
				W[y]=1
			elif W[int(y)]<0:
				W[y]=0

			acu, red, actual_tasa_agr = evaluate(W,_train,_y_train)
			if(actual_tasa_agr > mejor_tasa_agr):
				mejor_tasa_agr = actual_tasa_agr
				#Reseteo la cantidad de explorados sin mejorar
				_cantExplorados = 0
			else:
				 W[y] = W_anterior

	return W

def ejecutarBL(_data, _target):
	_inicio = current_milli_time()
	_time_actual = _inicio
	#Realizo las 5 particiones con mezcla y semilla
	skf = StratifiedKFold(n_splits=5, shuffle = True,  random_state = SEMILLA)
	skf.get_n_splits(_data, _target)
	#recorro cada una de las
	sumatoriaClas = 0
	sumatoriaRed = 0
	sumatoriaAgregada = 0
	_cont = 1
	tabla = BeautifulTable()
	tabla.column_headers = ["Particion","%cls","%redu","%agr", "tiempo"]
	for _train_index, _test_index in skf.split(_data, _target):
		#obtengo las particiones de test y train
		X_train, _X_test = _data[_train_index], _data[_test_index]
		y_train, _y_test = _target[_train_index], _target[_test_index]
		#obtengo los pèsos para esta particion
		_W  = BL(X_train, y_train)

		#obtengo las tasas  de evaular la solución
		tasa_clas, tasa_red, tasa_agr = evaluate(_W,_X_test,_y_test)
		time_anterior= _time_actual
		_time_actual =current_milli_time()
		sumatoriaClas += tasa_clas
		sumatoriaRed += tasa_red
		sumatoriaAgregada += tasa_agr
		tabla.append_row([_cont, tasa_clas, tasa_red,tasa_agr,( (_time_actual - time_anterior) / 1000.0)])
		_cont += 1

	_final = current_milli_time()
	#calculo el tiempo de ejecucion
	diferencia =(_final- _inicio) / 1000.0
	#muestro los estadisticos
	tabla.append_row(["Media",sumatoriaClas/5,sumatoriaRed/5,sumatoriaAgregada/5,(diferencia/5)])
	print(tabla)




TEXTURE = "ConjuntosDatos/texture.arff"
COLPOSCOPY = "ConjuntosDatos/colposcopy.arff"
IONOSPHERE = "ConjuntosDatos/ionosphere.arff"

dataC, targetC = cargarDatos(COLPOSCOPY,'int')
dataT, targetT = cargarDatos(TEXTURE,'int')
dataI, targetI = cargarDatos(IONOSPHERE,'str')

#relief
print("--------------------Ejecucion Knn--------------------")
print("--------------------TEXTURE--------------------")
ejecutarKNN(dataT, targetT)
print("------------------COLPOSCOPY----------------------")
ejecutarKNN(dataC, targetC)
print("--------------------IONOSPHERE--------------------")
ejecutarKNN(dataI, targetI)
input("Pulsa  Enter para continuar a la busqueda con relief...")

print("--------------------Ejecucion Relief--------------------")
print("--------------------COLPOSCOPY--------------------")
ejecucionRelief(dataC, targetC)
print("--------------------TEXTURE--------------------")
ejecucionRelief(dataT, targetT)
print("--------------------IONOSPHERE--------------------")
ejecucionRelief(dataI, targetI)
input("Pulsa  Enter para continuar a la busqueda local...")

print("--------------------Ejecucion BL--------------------")
print("--------------------COLPOSCOPY--------------------")
ejecutarBL(dataC, targetC)
print("--------------------TEXTURE--------------------")
ejecutarBL(dataT, targetT)
print("--------------------IONOSPHERE--------------------")
ejecutarBL(dataI, targetI)

