using Images, Colors, FFTW, StatsBase

## INSTRUCCIONES
# Para comprimir un archivo debe correrse la funcion guardar(nombre, direccion, quant)
# nombre: Nombre de archivo a comprimir, debe estar en la misma carpeta que el archivo de 
# julia
# direccion: nombre del archivo en el cual se guardara la imagen comprimida, por default 
# estara "ImagenComprimida.ext"
# quant: matriz de cuantizacion para realizar la compresion, por default estara la matriz 
# que nos dieron en la consigna del trabajo (la matriz debe estar en tipo Int64)

# para abrir la imagen simplemente corre la funcion abrir(nombre)
# nombre: nombre del archivo comprimido a abrir, debe estar en la misma carpeta que 
# el archivo de julia

# Si quieren probar con algunas de las imagenes que incluimos en la carpeta 
# guardar("paisaje.bmp")
# abrir("ImagenComprimida.ext")

#CONCLUSIONES al final del archivo



function multiplo16(im)
    #Para poder procesar la imagen con el algoritmo propuesto necesitamos que 
    #las dimensiones de la imagen sean múltiplo de 16. Por lo tanto, comenzamos 
    #con una función que calcula las dimensiones actuales y rellena simétricamente 
    #a izquierda y a derecha con píxeles negros.
	w, h = size(im)
	faltan_w = [Int(floor((16 - rem(w, 16))/2)), Int(ceil((16 - rem(w, 16))/2))]
	faltan_h = [Int(floor((16 - rem(h, 16))/2)), Int(ceil((16 - rem(h, 16))/2))]
	newW = w + faltan_w[1] + faltan_w[2]	
	newH = h + faltan_h[1] + faltan_h[2]
	procesada = Matrix{RGB{N0f8}}
	procesada = fill(RGB{N0f8}(0, 0, 0), Int(newW), Int(newH))
	for i in (faltan_w[1] + 1:size(im)[1] + faltan_w[1])
		for j in (faltan_h[1] + 1:size(im)[2] + faltan_h[1])
			procesada[i,j] = im[i - faltan_w[1] ,j - faltan_h[1]]
		end
	end
	return procesada
end

function tamañoOriginal(im, n, m)
	#También hacemos la función inversa, que dada una función que hemos rellenado
	#con pixeles negros para que el tamaño sea múltiplo de 16, quitamos los bordes
	#para recuperar la imagen original.
	w, h = n, m
	sobran_w = [Int(floor((16 - rem(w, 16))/2)), Int(ceil((16 - rem(w, 16))/2))]
	sobran_h = [Int(floor((16 - rem(h, 16))/2)), Int(ceil((16 - rem(h, 16))/2))]
	newW = w
	newH = h
	terminada = Matrix{RGB{N0f8}}
	terminada = fill(RGB{N0f8}(0, 0, 0), Int(newW), Int(newH))
	for i in (1:n)
		for j in (1:m)
			terminada[i,j] = im[i + sobran_w[1],j + sobran_h[1]]
		end
	end
	return terminada
end

function separacionYCbCr(im)
	#Descomponemos los tres canales de la imagen 
    #para trabajar sobre ellos por separado usando el comando channelview
	im_16 = multiplo16(im)
	im_ycbcr = YCbCr.(im_16)	
	im_YCbCr = channelview(im_ycbcr)
	Y = im_YCbCr[1,:,:] 
	Cb = im_YCbCr[2,:,:] 
	Cr = im_YCbCr[3,:,:]
	return Y, Cb, Cr
end

function matrizPromedio(im)
    #Creamos una función para reducir las matrices Cb y Cr promediando cada pixel con 
    #sus 4 pixels vecinos, y luego en la función centrar restamos 128 para que todos 
    #los valores se encuentren alrededor del 0.
	n = size(im)[1]		
	m = size(im)[2]
	nueva = Matrix{Float64}
	nueva = fill(Float64(0), Int(n/2), Int(m/2))
	#creamos una nueva matriz cuyas dimensiones son la mitad de la anterior
	for i in (2:2:n)
		for j in (2:2:m)
			nueva[Int(i/2), Int(j/2)] = (im[i, j] + im[i-1, j] + im[i, j-1] + im[i-1, j-1])/4
		end
	end
	return nueva
	#calculamos el promedio con los 4 píxeles vecinos y lo vamos guardando en la matriz nueva,
    #luego la retornamos
end

function centrar(Y, Cb, Cr)
	Cb_centrada = Cb .- 128
	Y_centrada = Y .- 128
	Cr_centrada = Cr .- 128
	return Y_centrada, Cb_centrada, Cr_centrada
end

function matrizPromedioInversa(Y, Cb, Cr)
    #Creamos también una función que realiza el proceso inverso a matrizPromedio y centrar,
    #tomando las matrices Y, Cb y Cr y devolviendo la imagen RGB.
	m = size(Cb)[2]
	n = size(Cb)[1]
	Y = Y.+128
	Cb = Cb.+128
	Cr = Cr.+128
	#des-centramos cada matriz sumando 128, consiguiendo que los valores estén entre 0 y 256
	nuevaCb = Matrix{Float64}
	nuevaCb = fill(Float64(0), n*2, m*2)
	nuevaCr = Matrix{Float64}
	nuevaCr = fill(Float64(0), n*2, m*2)
	#creamos las nuevas matrices Cb y Cr descomprimiendo las recibidas, que tenían la mitad 
    #del tamaño original
	for i in (2:2:2*n)
		for j in (2:2:2*m )
			i_ = Int(i/2)
			j_ = Int(j/2)
			nuevaCb[i, j] = Cb[i_, j_] 
			nuevaCb[i-1, j] = Cb[i_, j_] 
			nuevaCb[i-1, j-1] = Cb[i_, j_] 
			nuevaCb[i, j-1] = Cb[i_, j_]
			nuevaCr[i, j] = Cr[i_, j_]
			nuevaCr[i-1, j] = Cr[i_, j_]
			nuevaCr[i-1, j-1] = Cr[i_, j_]
			nuevaCr[i, j-1] = Cr[i_, j_]
		end
		#llenamos las nuevas matrices deshaciendo el promedio calculado en bloques de 4x4 
        #con valores iguales
	end
	A = [Y, nuevaCb, nuevaCr]
	nueva = colorview(YCbCr, A...) #transformamos el formato YCbCr a RGB
	return nueva
end

function transformada_cos(A)
    #Para cada matriz calculamos su transformada por bloques de 8x8
	n, m = size(A)
	transformada = fill(Float64(0), n, m)
	for i in (1:8:n) # i:i+7
		for j in (1:8:m) # j:j+7
			transformada[i:i+7,j:j+7] = dct(A[i:i+7,j:j+7])
		end
	end
	return transformada
end

function inversa_cos(A)
    #Hacemos también el proceso inverso: Para una matriz con la transformada por bloques
    #ya calculada, hacemos la transformada por bloques inversa para recuperar la matriz.
	n, m = size(A)
	destransformada = fill(Float64(0), n, m)
	for i in (1:8:n) # i:i+7
		for j in (1:8:m) # j:j+7
			destransformada[i:i+7,j:j+7] = idct(A[i:i+7,j:j+7])
		end
	end
	return destransformada
end

function cuantizacion(A, quant)
    #Descartamos frecuencias altas usando una matriz de cuantización
	n, m = size(A)
	nueva = fill(Int8(0), n, m)
	for i in (1:8:n) # i:i+7
		for j in (1:8:m) # j:j+7
			nueva[i:i+7,j:j+7] = round.(A[i:i+7,j:j+7] ./ quant)
            #para cuantizar la matriz tomamos cada bloque de 8x8 y lo dividimos
            #casillero a casillero por la matriz de cuantización 
		end
	end
	return nueva
end

function cuantizacionInversa(A, quant)
    #En el proceso inverso multiplicamos los bloques de 8x8 casillero a casillero por
    #la matriz de cuantización
	n, m = size(A)
	nueva = fill(Float64(0), n, m)
	for i in (1:8:n) # i:i+7
		for j in (1:8:m) # j:j+7
			nueva[i:i+7,j:j+7] = A[i:i+7,j:j+7] .* quant
		end
	end
	return nueva
end

function compresion(A)
	compresion = []
	n,m = size(A)
	for i in (1:8:n) # bloque eje x
		for j in (1:8:m)# bloque eje y
			sol = [[] for i in (1:8+8)]
            #leemos la matriz transformada en zigzag
			solucion = []
			for l in (1:8) # leemos adentro del bloque eje x
				for k in (1:8) # leemos adentro del bloque eje y
					sum = (l-1) + (k-1)
					if sum % 2 == 0
						pushfirst!(sol[sum + 1], A[i + (l-1), j + (k-1)])
					else
						push!(sol[sum + 1], A[i + (l-1), j + (k-1)])
                    end
				end
			end
			for i in sol
                #guardamos los elementos de la matriz leídos en zigzag
				for j in i
					push!(solucion, j)
				end
			end
			push!(compresion, rle(solucion))
            #comprimimos la solución con el método RLE y lo insertamos en la respuesta
		end
	end
	return compresion
end

#Procedimiento auxiliar para calcular la inversa de la compresion del bloque con el 
#orden de zigzag
    aux = [[] for i in (1:8+8)]
    posiciones = []
        
    for i in 1:8
        for j in 1:8
            if ((i-1) + (j-1)) % 2 == 0
               pushfirst!(aux[((i-1) + (j-1)) + 1], [i,j])
            else
                push!(aux[((i-1) + (j-1)) + 1], [i,j])
            end
        end
    end
    
        for i in aux
            for j in i
                push!(posiciones,j)
            end
        end
        

function reconstruccion_bloque(vals, reps)
	lista = inverse_rle(vals, reps)
	reconstruida = zeros(Int8,8,8)
	for l in 1:64
		i,j = posiciones[l]
		reconstruida[i,j] = lista[l]
	end
	return reconstruida
end

function descompresion(A,n,m)
    #La inversa de la función de compresión: toma un vector resultante de comprimir la
    #matriz, realiza el proceso RLE inverso y lee en zigzag inverso, retornando en
    #la matriz descomprimida.
	bloques = []
	for i in 1:length(A)
		val, rep = A[i]
		bloque = reconstruccion_bloque(val, rep)
		push!(bloques, bloque)
	end

	#Reconstruyamos ahora la matriz
	B = zeros(Int8, Int64(n), Int64(m)) 
	cntbloque = 1

	for i in (1:8:Int64(n))
		for j in (1:8:Int64(m))
			bloque_ = bloques[cntbloque]
			cntbloque += 1
			
			for k in (1:8)
				for l in (1:8)
					B[i + k - 1, j + l - 1] = bloque_[k,l]
				end
			end
		end 
	end
	return B
end

function abrir(nombre)
	#Dada una imagen realiza todo el proceso de descompresión usando las funciones
	#que armamos
	imagen = open(nombre)
	n = read(imagen, UInt16)
	m = read(imagen, UInt16)
	faltan_n = [Int(floor((16 - rem(n, 16))/2)), Int(ceil((16 - rem(n, 16))/2))]
	faltan_m = [Int(floor((16 - rem(m, 16))/2)), Int(ceil((16 - rem(m, 16))/2))]
	N = n + faltan_n[1] + faltan_n[2]	
	M = m + faltan_m[1] + faltan_m[2]
	#calculamos N y M para saber cuántos bloques tenemos
	quant_matrix = zeros(8, 8)
	for i in 1:8
		for j in 1:8
			quant_matrix[i, j] = read(imagen, UInt8)
		end
	end

	#Leemos el primer numero, lo mandamos a la primera posicion de vals, 
	#leemos el segundo numero, lo mandamos a la primera posicion de reps. 
	#Seguimos haciendo esto y agregando a vals y reps respectivamente.  
	#Cuando la cantidad de reps sea 64. Agregamos la tupla al vector compresion. 
	#Cuando la cantidad de tuplas sea M*N / 64. Pasamos a Cb y asi
	suma_reps = 0
	suma_tuplas = 0
	Y_comp = []
	while(suma_tuplas < N*M/64)
		suma_reps = 0
		val_aux = []
		rep_aux = []
		while(suma_reps < 64)
			val = read(imagen, Int8)
			rep = read(imagen, Int8)
			suma_reps += rep
			push!(val_aux, val)
			push!(rep_aux, rep)
		end
		rep_aux = convert(Array{Int64}, rep_aux)
		c = (val_aux, rep_aux)
		push!(Y_comp, c)
		suma_tuplas += 1
	end
	
	suma_tuplas = 0
	suma_reps = 0
	
	#Seguimos con Cb
	Cb_comp = []
	while(suma_tuplas < N*M/256)
		suma_reps = 0
		val_aux = []
		rep_aux = []
		while(suma_reps < 64)
			val = read(imagen, Int8)
			rep = read(imagen, Int8)
			suma_reps += rep
			push!(val_aux, val)
			push!(rep_aux, rep)
		end
		rep_aux = convert(Array{Int64}, rep_aux)
		c = (val_aux, rep_aux)
		push!(Cb_comp, c)
		suma_tuplas += 1
	end

	
	#seguimos con Cr
	suma_tuplas = 0
	suma_reps = 0
	Cr_comp = []
	while(suma_tuplas < N*M/256)
		suma_reps = 0
		val_aux = []
		rep_aux = []
		while(suma_reps < 64)
			val = read(imagen, Int8)
			rep = read(imagen, Int8)
			suma_reps += rep
			push!(val_aux, val)
			push!(rep_aux, rep)
		end
		rep_aux = convert(Array{Int64}, rep_aux)
		c = (val_aux, rep_aux)
		push!(Cr_comp, c)
		suma_tuplas += 1
	end

	
	# descomprimimos las matrices
	Y_desc = descompresion(Y_comp, N, M)
	Cb_desc = descompresion(Cb_comp, N/2, M/2)
	Cr_desc = descompresion(Cr_comp, N/2, M/2)


	#luego, las descuantizamos
	Y_desquant = cuantizacionInversa(Y_desc, quant)
	Cb_desquant = cuantizacionInversa(Cb_desc, quant)
	Cr_desquant = cuantizacionInversa(Cr_desc, quant)

	#ahora hacemos la transformada inversa de cada una
	Y_trans_inv = inversa_cos(Y_desquant) 
	Cb_trans_inv = inversa_cos(Cb_desquant) 
	Cr_trans_inv = inversa_cos(Cr_desquant)
	
	#las unimos y las convertimos a formato RGB
	definitiva_ = matrizPromedioInversa(Y_trans_inv, Cb_trans_inv, Cr_trans_inv)
	definitiva = tamañoOriginal(definitiva_, n, m)
	close(imagen)
	return definitiva
end

function guardar(nombre, direccion = "ImagenComprimida.ext", quant=quant)
	#Dada una imagen, la comprimimos usando las funciones y la guardamos.
	im = load(nombre)
	n = size(im)[1]
	m = size(im)[2]
	multiplo16(im)
	#separo las matrices y las prepar para transformar
	Y, Cb, Cr = separacionYCbCr(im)
	Cb_ = matrizPromedio(Cb)
	Cr_ = matrizPromedio(Cr)
	Y_cent, Cb_cent, Cr_cent = centrar(Y, Cb_, Cr_)
	#tramsformo las 3 matrices
	Cb_transformada = transformada_cos(Cb_cent)
	Cr_transformada = transformada_cos(Cr_cent)
	Y_transformada = transformada_cos(Y_cent)
	#realizo la cuantizacion
	Cb_cuant = cuantizacion(Cb_transformada, quant)
	Cr_cuant = cuantizacion(Cr_transformada, quant)
	Y_cuant = cuantizacion(Y_transformada, quant)
	#cambiamos los tipos de los numeros
	Cb_cuant_ = convert(Matrix{Int8}, Cb_cuant)
	Cr_cuant_ = convert(Matrix{Int8}, Cr_cuant)
	Y_cuant_ = convert(Matrix{Int8}, Y_cuant)
	n = convert(UInt16,n)
	m = convert(UInt16,m)
	#y luego la compresion
	Cb_comp_ = compresion(Cb_cuant_)
	Cr_comp_ = compresion(Cr_cuant_)
	Y_comp_ = compresion(Y_cuant_)
	#creamos el archivo
	io = open(direccion,"w")
	write(io,n)
	write(io,m)
	#escribimos la matriz de cuantizacion usada en el archivo
	for i in range(1, size(quant)[1])
		for j in range(1, size(quant)[2])
			write(io, UInt8(quant[i, j]))
		end
	end
	#ponemos repeticiones y luego los valores de las matrices comprimidas(Y, Cb, Cr)
	for i in 1: size(Y_comp_)[1]
		for j in range(1, size(Y_comp_[i][1])[1])
		write(io, Int8(Y_comp_[i][1][j]))
		write(io, Int8(Y_comp_[i][2][j]))
		end
	end
	for i in range(1, size(Cb_comp_)[1])
		for j in range(1, size(Cb_comp_[i][1])[1])
		write(io, Int8(Cb_comp_[i][1][j]))
		write(io, Int8(Cb_comp_[i][2][j]))
		end
	end
	for i in range(1, size(Cr_comp_)[1])
		for j in range(1, size(Cr_comp_[i][1])[1])
		write(io, Int8(Cr_comp_[i][1][j]))
		write(io, Int8(Cr_comp_[i][2][j]))
		end
	end
	close(io)
end

quant=[16 11 10 16 24 40 51 61;
		12 12 14 19 26 58 60 55;
		14 13 16 24 40 57 69 56;
		14 17 22 29 51 87 80 62;
		18 22 37 56 68 109 103 77;
		24 35 55 64 81 104 113 92;
		49 64 78 87 103 121 120 101;
		72 92 95 98 112 100 103 99]
		#matriz de cuantización que usamos

quant2 = quant*2

#Conclusiones:
# Con la imagen paisaje.bmp, la cual tenia un tamaño original de 7.2mb, al comprimirla con 
# la matriz de cuantizacion dada en el TP logramos comprimir la imagen a 1.2mb, y a ojo 
# humano, no se logran ver diferencias con la imagen original luego de abrirla.
# Ademas, probamos con una matriz de cuantizacion que era exactamente el doble que la 
# matriz anterior, logrando comprimir la imagen a 700KB lo cual es razonable, ya que al 
# incrementar los valores de la matriz de cuantizacion mas numeros seran llevados a ceros, 
# pero esto tambien implico que la imagen se veia con un tono mas grisaceo a ojo humano 
# luego de abrirla.
