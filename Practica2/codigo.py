#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 17:46:32 2019

@author: antonio
"""
#documento para el 1-knn http://www.sc.ehu.es/ccwbayes/docencia/mmcc/docs/t9knn.pdf
from math import sqrt, ceil
from random import randint
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

SEMILLA = 1231
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


def BLX( cruce1, cruce2) :
    alfa = 0.3
    h1 = []
    h2 = []
    for i in range(cruce1.size):
        Cmax = max(cruce1[i],cruce2[i])
        Cmin = min(cruce1[i],cruce2[i])     
        I = Cmax - Cmin
        h1.append(np.random.uniform(Cmin-I*alfa, Cmax+I*alfa))
        h2.append(np.random.uniform(Cmin-I*alfa, Cmax+I*alfa))
    h1 = np.asarray(h1)
    h2 = np.asarray(h2)

    return h1,h2

def cx_arithmetic(ind1, ind2):
    alphas = 0.5
    c1 = (1 - alphas) * ind1 + alphas * ind2
    c2 = alphas * ind1 + (1 - alphas) * ind2
    ind1[:] = c1
    ind2[:] = c2
    return ind1, ind2

def AGGBLX(train, y_train):

    TAM_POBLACION = 30
    PRO_CRUCE = 0.7
    PRO_MUT = 0.01
    CANT_CRUCES = int(TAM_POBLACION*PRO_CRUCE)
    CANTMUTACIONES = int(TAM_POBLACION*PRO_MUT*train[0].size)

    pobla = []
    evalu = []
    #creo la poblacion y los evaluo a todos.
    for i in range(TAM_POBLACION):
        pobla.append( np.random.random(train[0].size))
        evalu.append(evaluate(pobla[i],train, y_train))
    pobla = np.asarray(pobla)
    
    MAX = 15000
    cantEjecuciones = 0
    mejor = np.random.random(train[0].size)
    while(cantEjecuciones < MAX):
        
        
        valmejor = [0,0,0];
        mejores = []
        for i in range(TAM_POBLACION):
            row_i = np.random.choice(pobla.shape[0],1)
            primero = pobla[row_i, :]
            primero = primero.reshape(-1,)
            agre_actual1 = evalu[row_i[0]]

            row_i = np.random.choice(pobla.shape[0],1)
            segundo = pobla[row_i, :] 
            segundo = segundo.reshape(-1,)
            agre_actual2 = evalu[row_i[0]]

            if(agre_actual1[2] > agre_actual2[2]):
                mejores.append(primero)
                if(agre_actual1[2] > valmejor[2]):
                    valmejor = agre_actual1
                    mejor = primero
            else:
                mejores.append(segundo)
                if(agre_actual2[2] > valmejor[2]):
                    valmejor = agre_actual2
                    mejor = segundo
                    
        pobla = np.asarray(mejores)
        
        sinmuta = pobla[CANT_CRUCES+1:]
        sinmuta = np.asarray(sinmuta)
        poblacion_cruzada = []
        for i in range(ceil(CANT_CRUCES/2)):
                uno,dos = BLX(pobla[i], pobla[i+1])    
                poblacion_cruzada.append(uno)
                poblacion_cruzada.append(dos)
        pobla = np.concatenate((poblacion_cruzada,sinmuta))        
        
        
        for i in range(CANTMUTACIONES):
            indiceCromo = randint(0,TAM_POBLACION-1)
            genAMutar = randint(0,train[0].size-1)
            Z =  np.random.normal(0.0, 0.3)
            pobla[indiceCromo][genAMutar] += Z
            if pobla[indiceCromo][genAMutar]  > 1:
                pobla[indiceCromo][genAMutar] = 1
            elif pobla[indiceCromo][genAMutar]  < 0:
                pobla[indiceCromo][genAMutar] = 0
               
      
        peor = [100,100,100]
        indicePeor = 0
        esta = False
        for i in range(pobla.shape[0]):
            cantEjecuciones+=1		
            actual = evaluate(pobla[i],train, y_train)
            evalu[i] = actual
            if(np.equal(mejor, pobla[i])[0]):
                esta = True
                break

            if peor[2] > actual[2]:
                peor = actual
                indicePeor = i
        if esta == False:
            pobla[indicePeor] = mejor 
            evalu[indicePeor] = evaluate(mejor,train, y_train)
        
    mejor = [0,0,0]
    indiceMejor = 0
    for i in range(pobla.shape[0]):
        actual =evaluate(pobla[i],train, y_train)
        if actual[2] > mejor[2]:
            mejor = actual
            indiceMejor = i
    return pobla[indiceMejor]


def AGGCA(train, y_train):

    TAM_POBLACION = 30
    PRO_CRUCE = 0.7
    PRO_MUT = 0.01
    CANT_CRUCES = int(TAM_POBLACION*PRO_CRUCE)
    CANTMUTACIONES = int(TAM_POBLACION*PRO_MUT*train[0].size)

    pobla = []
    evalu = []
    #creo la poblacion y los evaluo a todos.
    for i in range(TAM_POBLACION):
        pobla.append( np.random.random(train[0].size))
        evalu.append(evaluate(pobla[i],train, y_train))
    pobla = np.asarray(pobla)
    
    MAX = 15000
    cantEjecuciones = 0
    mejor = np.random.random(train[0].size)
    while(cantEjecuciones < MAX):
        
        
        valmejor = [0,0,0];
        mejores = []
        for i in range(TAM_POBLACION):
            row_i = np.random.choice(pobla.shape[0],1)
            primero = pobla[row_i, :]
            primero = primero.reshape(-1,)
            agre_actual1 = evalu[row_i[0]]

            row_i = np.random.choice(pobla.shape[0],1)
            segundo = pobla[row_i, :] 
            segundo = segundo.reshape(-1,)
            agre_actual2 = evalu[row_i[0]]

            if(agre_actual1[2] > agre_actual2[2]):
                mejores.append(primero)
                if(agre_actual1[2] > valmejor[2]):
                    valmejor = agre_actual1
                    mejor = primero
            else:
                mejores.append(segundo)
                if(agre_actual2[2] > valmejor[2]):
                    valmejor = agre_actual2
                    mejor = segundo
                    
        pobla = np.asarray(mejores)
        np.random.shuffle(pobla)
        sinmuta = pobla[CANT_CRUCES+1:]
        sinmuta = np.asarray(sinmuta)
        poblacion_cruzada = []
        for i in range(ceil(CANT_CRUCES/2)):
                uno,dos = cx_arithmetic(pobla[i], pobla[i+1])    
                poblacion_cruzada.append(uno)
                poblacion_cruzada.append(dos)
        pobla = np.concatenate((poblacion_cruzada,sinmuta))        
        
        
        for i in range(CANTMUTACIONES):
            indiceCromo = randint(0,TAM_POBLACION-1)
            genAMutar = randint(0,train[0].size-1)
            Z =  np.random.normal(0.0, 0.3)
            pobla[indiceCromo][genAMutar] += Z
            if pobla[indiceCromo][genAMutar]  > 1:
                pobla[indiceCromo][genAMutar] = 1
            elif pobla[indiceCromo][genAMutar]  < 0:
                pobla[indiceCromo][genAMutar] = 0
               

        peor = [100,100,100]
        indicePeor = 0
        esta = False
        for i in range(pobla.shape[0]):
            cantEjecuciones+=1		
            actual = evaluate(pobla[i],train, y_train)
            evalu[i] = actual
            if(np.equal(mejor, pobla[i])[0]):
                esta = True
                break

            if peor[2] > actual[2]:
                peor = actual
                indicePeor = i
        if esta == False:
            pobla[indicePeor] = mejor 
            evalu[indicePeor] = evaluate(mejor,train, y_train)
        
    mejor = [0,0,0]
    indiceMejor = 0
    for i in range(pobla.shape[0]):
        actual =evaluate(pobla[i],train, y_train)
        if actual[2] > mejor[2]:
            mejor = actual
            indiceMejor = i
    return pobla[indiceMejor]


def AGEBLX(train, y_train):

    TAM_POBLACION = 30
    MEJORES = 2
    PRO_MUT = 0.001
    CANTMUTACIONES = int(MEJORES*PRO_MUT*train[0].size)

    pobla = []
    evalu = []
    #creo la poblacion y los evaluo a todos.
    for i in range(TAM_POBLACION):
        pobla.append( np.random.random(train[0].size))
        evalu.append(evaluate(pobla[i],train, y_train))
    pobla = np.asarray(pobla)
    
    MAX = 15000
    cantEjecuciones = 0
    mejor = np.random.random(train[0].size)
    while(cantEjecuciones < MAX):
        
        
        valmejor = [0,0,0];
        mejores = []
        for i in range(MEJORES):
            row_i = np.random.choice(pobla.shape[0],1)
            primero = pobla[row_i, :]
            primero = primero.reshape(-1,)
            agre_actual1 = evalu[row_i[0]]

            row_i = np.random.choice(pobla.shape[0],1)
            segundo = pobla[row_i, :] 
            segundo = segundo.reshape(-1,)
            agre_actual2 = evalu[row_i[0]]

            if(agre_actual1[2] > agre_actual2[2]):
                mejores.append(primero)
                if(agre_actual1[2] > valmejor[2]):
                    valmejor = agre_actual1
                    mejor = primero
            else:
                mejores.append(segundo)
                if(agre_actual2[2] > valmejor[2]):
                    valmejor = agre_actual2
                    mejor = segundo
                    
        mejores = np.asarray(mejores)
        
       
        uno,dos = BLX(mejores[0], mejores[1])    

        uno = uno.reshape(1,uno.size)
        dos = dos.reshape(1,dos.size)

        mejores = np.concatenate((uno,dos))
        
        for i in range(CANTMUTACIONES):
            indiceCromo = randint(0,MEJORES-1)
            genAMutar = randint(0,train[0].size-1)
            Z =  np.random.normal(0.0, 0.3)
            mejores[indiceCromo][genAMutar] += Z
            if mejores[indiceCromo][genAMutar]  > 1:
                mejores[indiceCromo][genAMutar] = 1
            elif mejores[indiceCromo][genAMutar]  < 0:
                mejores[indiceCromo][genAMutar] = 0
               
      
        peor = [100,100,100]
        indicePeor = 0
        for i in range(TAM_POBLACION):
            cantEjecuciones+=1
            actual = evaluate(pobla[i],train, y_train)
            evalu[i] = actual

            if peor[2] > actual[2]:
                peor = actual
                indicePeor = i
        pobla = np.delete(pobla, indicePeor, axis=0)
        peor = [100,100,100]
        for i in range(TAM_POBLACION-1):
            cantEjecuciones+=1		
            actual = evaluate(pobla[i],train, y_train)
            evalu[i] = actual
            if peor[2] > actual[2]:
                peor = actual
                indicePeor = i               
        pobla = np.delete(pobla, indicePeor, axis=0)
        
        pobla = np.concatenate((pobla,uno))
        pobla = np.concatenate((pobla,dos))  
        
    mejor = [0,0,0]
    indiceMejor = 0
    for i in range(pobla.shape[0]):
        actual =evaluate(pobla[i],train, y_train)
        if actual[2] > mejor[2]:
            mejor = actual
            indiceMejor = i
    return pobla[indiceMejor]

def ejecutarAGGBLX(_data, _target):
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
		_W  = AGGBLX(X_train, y_train)

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
    
def ejecutarAGGCA(_data, _target):
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
		_W  = AGGCA(X_train, y_train)

		#obtengo las tasas  de evaular la solución
		tasa_clas, tasa_red, tasa_agr = evaluate(_W,_X_test,_y_test)
		time_anterior= _time_actual
		_time_actual =current_milli_time()
		sumatoriaClas += tasa_clas
		sumatoriaRed += tasa_red
		sumatoriaAgregada += tasa_agr
		print(str(tasa_clas)+"-" +str(tasa_red)+" "+str(tasa_agr))
		tabla.append_row([_cont, tasa_clas, tasa_red,tasa_agr,( (_time_actual - time_anterior) / 1000.0)])
		_cont += 1

	_final = current_milli_time()
	#calculo el tiempo de ejecucion
	diferencia =(_final- _inicio) / 1000.0
	#muestro los estadisticos
	tabla.append_row(["Media",sumatoriaClas/5,sumatoriaRed/5,sumatoriaAgregada/5,(diferencia/5)])
	print(tabla)
    
    
def ejecutarAGEBLX(_data, _target):
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
		_W  = AGEBLX(X_train, y_train)

		#obtengo las tasas  de evaular la solución
		tasa_clas, tasa_red, tasa_agr = evaluate(_W,_X_test,_y_test)
		time_anterior= _time_actual
		_time_actual =current_milli_time()
		sumatoriaClas += tasa_clas
		sumatoriaRed += tasa_red
		sumatoriaAgregada += tasa_agr
		print(str(tasa_clas)+"-" +str(tasa_red)+" "+str(tasa_agr))
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
'''
print("--------------------Ejecucion Knn--------------------")
print("------------------COLPOSCOPY----------------------")
ejecutarKNN(dataC, targetC)
print("--------------------TEXTURE--------------------")
ejecutarKNN(dataT, targetT)
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
'''

print("--------------------Ejecucion AGE-BLX--------------------")
print("--------------------COLPOSCOPY--------------------")
ejecutarAGEBLX(dataC, targetC)
print("--------------------TEXTURE--------------------")
ejecutarAGEBLX(dataT, targetT)
print("--------------------IONOSPHERE--------------------")
ejecutarAGEBLX(dataI, targetI)

print("--------------------Ejecucion AGG-CA--------------------")
print("--------------------COLPOSCOPY--------------------")
ejecutarAGGCA(dataC, targetC)
print("--------------------TEXTURE--------------------")
ejecutarAGGCA(dataT, targetT)
print("--------------------IONOSPHERE--------------------")
ejecutarAGGCA(dataI, targetI)

print("--------------------Ejecucion AGG-BLX--------------------")
print("--------------------COLPOSCOPY--------------------")
ejecutarAGGBLX(dataC, targetC)
print("--------------------TEXTURE--------------------")
ejecutarAGGBLX(dataT, targetT)
print("--------------------IONOSPHERE--------------------")
ejecutarAGGBLX(dataI, targetI)

