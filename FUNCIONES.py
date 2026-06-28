# = = = = = = = = = = = = = PAQUETES = = = = = = = = = = = = = #
import healpy as hp
import numpy as np
import scipy.stats as scs
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import camb as camb
from camb import model, initialpower
from math import pi
from scipy.special import eval_legendre, legendre
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

def HISTOGRAMA_G(datos, titulo, rango_x, rango_y, resolucion, mu_manual = None, sigma_manual = None):
    'Nos permite obtener un histograma y su ajuste gaussiano'
    'Permite obtener un ajuste manual si se proporcionan los parámetros de media y desviación típica'
   
    # Tomamos primero los datos de μ, σ y κ.
    mu_python = np.mean(datos)
    sigma_python = np.std(datos)
    kurtosis = scs.kurtosis(datos, fisher=True)
    asimetria = scs.skew(datos)

    mu = mu_python
    sigma = sigma_python

    ancho_bin = (rango_x[1] - rango_x[0]) / resolucion

    # Lo siguiente que hacemos es generar un histograma con estos datos, donde hemos especificado el 
    # rango del eje x y la resolución de este.
    plt.hist(datos, bins=np.arange(rango_x[0], rango_x[1], ancho_bin), color='r', alpha=0.2, density=True)
    x = np.linspace(rango_x[0], rango_x[1], 10000)

    # Sobre este histograma, dibujamos la curva gaussiana que obtenemos con los datos tomados al
    # principio del programa.
    curva_gaussiana = scs.norm.pdf(x, mu, sigma)
    plt.plot(x, curva_gaussiana, 'k-', linewidth=2, 
             label=f'Ajuste Gaussiano\nμ={mu:.2e}, σ={sigma:.2e}\n κ={kurtosis:.2f}, γ={asimetria:.2e}')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(mu, color='k', linestyle='--', linewidth=1.5)

    # AJUSTE MANUAL (si se proporciona alguno de los parámetros)
    if (mu_manual is not None) or (sigma_manual is not None):
        mu1 = mu_manual if mu_manual is not None else mu_python
        sigma1 = sigma_manual if sigma_manual is not None else sigma_python

        ajuste = scs.norm.pdf(x, mu1, sigma1)
        plt.plot(x, ajuste, 'b:', linewidth=1.2, 
                 label=f'Ajuste Manual\nμ={mu1:.2e}, σ={sigma1:.2e}')

    # Finalmente representamos todo junto.
    plt.xlim(rango_x)
    plt.ylim(rango_y)
    plt.xlabel(r'Temperatura ($\mu$K)', fontsize= 15)
    plt.ylabel(r'Densidad de Probabilidad ($\mu\text{K}^{-1}$)', fontsize=15)
    plt.title(titulo, fontsize = 18)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize = 13)
    


def COMPARACION(mapas):
    'Obtenemos una tabla con los distintos parámetros de ajuste de la distribución'
    # Generamos primero un vector vacío que iremos llenando con los distintos valores estadísticos.
    filas = []

    # En este bucle anexamos uno a uno los datos estadísticos asociados a cada mapa
    for nombre, datos in mapas.items():
        filas.append({
            'Mapa': nombre,
            'Curtosis (κ)': scs.kurtosis(datos, fisher=True),
            'Asimetría (γ)': scs.skew(datos),
            'Media (μ)': np.mean(datos),
            'Desviación típica (σ)': np.std(datos)
        })
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    return pd.DataFrame(filas)



def PARCHE(mapa, nside, norte=True, mask_gal=None):
    'Con esta función, aplicamos un parche sobre las zonas que no nos interesan, pudiendo separar'
    'por cuadrantes el mapa'

    # Obtenemos los ángulos esféricos (theta: colatitud, phi: longitud) de cada píxel
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # Generamos un array booleano inicial (True = píxel enmascarado)
    # En este caso, creamos máscaras según el hemisferio seleccionado
    if norte:
        # MÁSCARA HEMISFERIO NORTE (MHN)
        MH = np.ones(npix, dtype=bool)      # Array de unos con el mismo tamaño que el mapa
        MH[theta < np.pi/2] = 0             # Array donde si se cumple la condición, ese punto pasa de 1 a 0

        # MÁSCARA NORESTE (MHNE)
        ME = np.ones(npix, dtype=bool)
        ME[(theta < np.pi/2) & (phi < np.pi)] = 0

        # MÁSCARA NOROESTE (MHNO)
        MO = np.ones(npix, dtype=bool)
        MO[(theta < np.pi/2) & (phi > np.pi)] = 0

    else:
        # MÁSCARA HEMISFERIO SUR (MHS)
        MH = np.ones(npix, dtype=bool)
        MH[theta > np.pi/2] = 0

        # MÁSCARA SURESTE (MHSE)
        ME = np.ones(npix, dtype=bool)
        ME[(theta > np.pi/2) & (phi < np.pi)] = 0

        # MÁSCARA SUROESTE (MHSO)
        MO = np.ones(npix, dtype=bool)
        MO[(theta > np.pi/2) & (phi > np.pi)] = 0

    # Juntamos con la máscara del plano galáctico si existe
    if mask_gal is not None:
        MH = np.logical_not(mask_gal) | MH
        ME = np.logical_not(mask_gal) | ME
        MO = np.logical_not(mask_gal) | MO

    # Aplicamos las máscaras a los mapas
    mapa_H = hp.ma(mapa)
    mapa_H.mask = MH

    mapa_E = hp.ma(mapa)
    mapa_E.mask = ME

    mapa_O = hp.ma(mapa)
    mapa_O.mask = MO

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    return mapa_H, mapa_E, mapa_O



def ERRORGAUSS(datos, titulo, rango_x, rango_y, resolucion, mu_manual=None, sigma_manual=None):
    'Esta función se encarga de obtener visualmente y numéricamente la diferencia entre' 
    'la curva experimental y el ajuste gaussiano'

    # Tomamos primero los datos de μ, σ y κ.
    mu = np.mean(datos)
    sigma = np.std(datos)
    kurtosis = scs.kurtosis(datos, fisher=True)   

    ancho_bin = (rango_x[1] - rango_x[0]) / resolucion

    # Obtenemos los datos del histograma
    hist, edges = np.histogram(datos, bins=np.arange(rango_x[0], rango_x[1], ancho_bin), density=True)

    # Calculamos los centros de cada bin, de forma que tengamos la posición de estos.
    centro_bin = 0.5 * (edges[:-1] + edges[1:])

    # Realizamos el ajuste gaussiano automático sobre estos centros
    ajuste_gaussiano = scs.norm.pdf(centro_bin, mu, sigma)

    # Restamos a los datos experimentales (histograma), la curva del ajuste automático.
    diferencia = hist - ajuste_gaussiano

    ss_res = np.sum((hist - ajuste_gaussiano) ** 2)
    ss_tot = np.sum((hist - np.mean(hist)) ** 2)
    semejanza = 1 - (ss_res / ss_tot)

    # Dibujamos ahora las barras con la diferencia automática
    plt.bar(centro_bin, diferencia, width=ancho_bin, color='r', alpha=0.2,
            label=f'Diferencia, $R^2$ = {semejanza:.4f}')

    # Si se pasa un ajuste manual, dibujamos la diferencia como línea sobre las barras
    if (mu_manual is not None) or (sigma_manual is not None):
        mu1 = mu_manual if mu_manual is not None else mu
        sigma1 = sigma_manual if sigma_manual is not None else sigma

        ajuste_manual = scs.norm.pdf(centro_bin, mu1, sigma1)
        diferencia_manual = hist - ajuste_manual

        # Dibujamos la diferencia manual como línea escalonada
        plt.plot(centro_bin, diferencia_manual, 'r:', drawstyle='steps-mid', linewidth=1.2,
                 label=f'Ajuste Manual\nμ={mu1:.2e}, σ={sigma1:.2e}\nSemejanza={semejanza:.4e}')

    # Ajustes del plot
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axvline(mu, color='k', linestyle='--', linewidth=1.5)
    plt.axhline(0)

    plt.xlim(rango_x)
    plt.ylim(rango_y)
    plt.title(titulo, fontsize = 18)
    plt.xlabel(r'Temperatura ($\mu$K)', fontsize= 15)
    plt.ylabel(r'Diferencia (Hist. $-$ Ajuste) ($\mu\text{K}^{-1}$)', fontsize=13)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize = 13)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    return diferencia


    
def CORRELACION(mapa1, mapa2, titulo, rango_x, rango_y, resolucion, mostrar=True):
    'Compara dos mapas usando gaussianas y muestra la diferencia'

    mu1, sigma1 = np.mean(mapa1), np.std(mapa1)
    mu2, sigma2 = np.mean(mapa2), np.std(mapa2)

    # Obtenemos el ancho de bin
    ancho_bin = (rango_x[1] - rango_x[0]) / resolucion

    # Array para evaluar la gaussiana
    x = np.arange(rango_x[0], rango_x[1], ancho_bin)

    # Gaussianas
    gaussiana1 = scs.norm.pdf(x, mu1, sigma1)
    gaussiana2 = scs.norm.pdf(x, mu2, sigma2)

    # Diferencia y suma
    diferencia_gauss = gaussiana1 - gaussiana2
    suma_gauss = gaussiana1 + gaussiana2

    # Correlación global
    correlacion = 1 - np.sum(np.abs(diferencia_gauss)) / np.sum(suma_gauss) if np.sum(suma_gauss) != 0 else 0

    # Gráfico
    if mostrar:
        plt.plot(x, diferencia_gauss, 'k-', linewidth=1.5, label='Diferencia de gaussianas')
        plt.axvline(mu1, color='b', linestyle='--', linewidth=1.5,
                    label=f'mapa1 μ={mu1:.2e}, σ={sigma1:.2e}')
        plt.axvline(mu2, color='r', linestyle='--', linewidth=1.5,
                    label=f'mapa2 μ={mu2:.2e}, σ={sigma2:.2e}')
        plt.axhline(0, color='k', linewidth=1)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(rango_x)
        plt.ylim(rango_y)
        plt.title(titulo)
        plt.xlabel(r'Temperatura ($\mu$K)')
        plt.ylabel('Diferencia de gaussianas')
        plt.legend()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - #    
    return correlacion



def FUNCION_CORRELACION_2P(Cl, min_ell, max_ell, array_angulos_rad, titulo, mostrar=True):
    'Nos permite obtener la función de correlación a dos puntos dados los coeficientes Cl y los valores mínimo y máximo de ell.'
    'Contiene además una opción para mostrar la gráfica.'

    # Iniciamos primero el array que contendrá los distintos C_theta:
    corr = []
    for t in array_angulos_rad:
        c_theta = 0
        cost = np.cos(t)
        for l in range(min_ell, max_ell + 1):
            c_theta += (2*l + 1)*Cl[l]*eval_legendre(l, cost)
        C_Theta = c_theta/(4*pi)
        corr.append(C_Theta)

    corr = np.array(corr)

    if mostrar == True:
        # Para representar, es más visual si utilizamos ángulos en grados en vez de radianes.
        array_angulos_deg = np.rad2deg(array_angulos_rad)

        # Elegimos un estilo bonito de gráfica 
        plt.style.use('seaborn-v0_8-whitegrid')

        plt.figure(figsize=(14,10), dpi=120)
        plt.plot(array_angulos_deg, corr, color='navy', lw=2.5, antialiased=True)

        plt.xlabel(r"Ángulo $[^\circ]$", fontsize=18)
        plt.ylabel(r"$C(\theta)$ [μK²]", fontsize=18)
        plt.title(titulo, fontsize=22, pad=15)

        # Márgenes
        plt.margins(x=0.02, y=0.05)

        # Hacemos el grid más sutil
        plt.minorticks_on()
        plt.grid(True, which='major', ls='--', lw=0.7, alpha=0.6)
        plt.grid(True, which='minor', ls=':', lw=0.4, alpha=0.3)

        # Ejes
        plt.tick_params(axis='both', which='major', labelsize=14)

        # Línea de referencia en cero
        plt.axhline(0, color='black', lw=1, alpha=0.6)

        # Línea vertical a 60 grados (para observar la anisotropía a grandes ángulos)
        plt.axvline(60, color='gray', lw=1, ls='--', alpha=0.6)

        # Ticks del eje x cada 20 grados
        plt.xticks(np.arange(0, 181, 20))

        plt.tight_layout()
        plt.show() 

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    return corr



def PLOT_COMPARACION_FC2P(lista_corr, array_angulos_rad, titulo, lista_nombre_mapa):
    'Nos permite representar en un único plot distintas funciones de correlación ya calculadas.'

    # Para representar, es más visual si utilizamos ángulos en grados en vez de radianes.
    array_angulos_deg = np.rad2deg(array_angulos_rad)

    # Elegimos un estilo  de gráfica 
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(14,10), dpi=120)

    # Dibujamos las curvas indicadas en la lista de correlaciones de cada mapa
    for corr, nombre in zip(lista_corr, lista_nombre_mapa):
        plt.plot(array_angulos_deg, corr, lw=2.5, antialiased=True, label=nombre)

    plt.xlabel(r"Ángulo $[^\circ]$", fontsize=18)
    plt.ylabel(r"$C(\theta)$ [μK²]", fontsize=18)
    plt.title(titulo, fontsize=22, pad=15)

    # Márgenes y grid
    plt.margins(x=0.02, y=0.05)
    plt.minorticks_on()
    plt.grid(True, which='major', ls='--', lw=0.7, alpha=0.6)
    plt.grid(True, which='minor', ls=':', lw=0.4, alpha=0.3)

    # Ejes
    plt.tick_params(axis='both', which='major', labelsize=14)

    # Línea de referencia en cero
    plt.axhline(0, color='black', lw=1, alpha=0.6)

    # Línea vertical a 60 grados
    plt.axvline(60, color='gray', lw=1, ls='--', alpha=0.6)

    # Ticks del eje x cada 20 grados
    plt.xticks(np.arange(0, 181, 20))

    plt.legend(fontsize=20)

    plt.tight_layout()
    plt.show()



def ASIMETRIA_NS(mapa, mask_gal):
    'Calcula la asimetría entre dos hemisferios de un mapa de CMB'
    
    # Obtenemos los ángulos esféricos (theta: colatitud, phi: longitud) de cada píxel
    npix = hp.nside2npix(NSIDE)
    theta, phi = hp.pix2ang(NSIDE, np.arange(npix))

    # Aplicamos las máscaras de hemisferio
    MN = np.ones(npix, dtype=bool)        # Array de unos con el mismo tamaño que el mapa
    MN[theta < np.pi/2] = 0               # Array donde si se cumple la condición, ese punto pasa de 1 a 0

    MS = np.ones(npix, dtype=bool)        # Array de unos con el mismo tamaño que el mapa
    MS[theta > np.pi/2] = 0               # Array donde si se cumple la condición, ese punto pasa de 1 a 0

    # Si hemos especificado la máscara galáctica, la juntamos con las máscaras de los 
    # distintos hemisferios
    if mask_gal is not None:
        MN = np.logical_not(mask_gal) | MN
        MS = np.logical_not(mask_gal) | MS

    # Cargamos ahora un mapa para cada hemisferio y le aplicamos
    # la correspondiente máscara
    mapa_norte = hp.ma(mapa)              
    mapa_norte.mask = MN        
    
    mapa_sur = hp.ma(mapa)                
    mapa_sur.mask = MS      

    mapa_total = hp.ma(mapa)
    mapa_total.mask = np.logical_not(mask_gal)

    # Lo siguiente es obtener las varianzas de los tres mapas: norte, sur y total

    sigma_norte = np.var(mapa_norte.compressed())
    sigma_sur = np.var(mapa_sur.compressed())
    sigma_total = np.var(mapa_total.compressed())                 
    
    # Finalmente escribimos la expresión de la asimetría
    asimetria = (sigma_norte - sigma_sur)/sigma_total

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    return asimetria



def ASIMETRIA_PLANO(mapa, mask_gal, nside, theta, phi, titulo,mostrar = False):
    'Obtiene diversos planos que dividen en dos el mapa de CMB.'
    'Calcula la asimetría entre dos hemisferios de este mapa, para cada plano.'

    # Obtenemos los ángulos esféricos (theta: colatitud, phi: longitud) de cada píxel
    npix = hp.nside2npix(nside)
    theta_r, phi_r = hp.pix2ang(nside, np.arange(npix))
    
    # Aplicamos la máscara al mapa (si está definida).
    mapa_cargado = hp.ma(mapa)
    if mask_gal is not None:
        mapa_cargado.mask = np.logical_not(mask_gal)

    mapa_ajustado = hp.ud_grade(mapa_cargado, nside) 
      
    # Obtenemos el vector del plano
    x = np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    y = np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    z = np.cos(np.radians(theta))

    n = np.array([x, y, z])
    
    # Una vez obtenido el vector del plano, obtenemos los vectores
    # de posición r tal que al hacer el producto escalar de dichos vectores
    # con el del plano obtendremos si estamos situados en el "norte" o el 
    # "sur".
    r = np.column_stack([np.sin(theta_r) * np.cos(phi_r),
                         np.sin(theta_r) * np.sin(phi_r),
                         np.cos(theta_r)])

    # Para evitar usar .compressed() todo el rato, vamos a 
    # seleccionar directamente qué pixeles son válidos tras aplicar
    # la máscara. De esta forma los cálculos son más rápidos.
    if mask_gal is not None:
        mapa_valido = mapa_ajustado.data[np.logical_not(mapa_ajustado.mask)]
        r = r[np.logical_not(mapa_ajustado.mask), :]
    else:
        mapa_valido = mapa_ajustado.data

    sigma_total = np.var(mapa_valido) 

    # Hacemos ahora el producto escalar
    prod_escalar = r @ n

    # El "norte" será donde el producto sea positivo
    norte = (prod_escalar > 0)    

    # El "sur" para los puntos que no pertenecen al norte
    sur = ~norte   

    # Obtenemos finalmente la varianza del norte y del sur
    sigma_norte = np.var(mapa_valido[norte])
    sigma_sur = np.var(mapa_valido[sur])
    asimetria = (sigma_norte - sigma_sur) / sigma_total   

    if mostrar == True:
        # Pintamos primero el mapa de CMB sin alterar sus datos.
        hp.mollview(mapa_ajustado,
                    title = f'{titulo}, región norte sombreada \nθ={theta:.1f}°, φ={phi:.1f}° | A={asimetria:.5f}',
                    cmap = 'spring',
                    hold = False)

        # Trazamos ahora la línea del plano. Para ello, usamos que r * n ha de ser cero en el plano. Despejando
        # obtenemos una función para theta dependiente de phi. Si le damos los valores desde cero hasta 2pi a 
        # phi, obtendremos la línea del plano completa
        phi_linea = np.linspace(0, 2 * np.pi, 500)
        denominador = x * np.cos(phi_linea) + y * np.sin(phi_linea)
        theta_linea = np.arctan2(-z,denominador)
        
        # Usando projplot añadimos esta línea a los datos del mapa.
        hp.projplot(theta_linea, phi_linea, color='black', linewidth=2, linestyle='-')
        
        plt.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    return asimetria        



def FASES_LUNA(mapa, mask_gal, sub_theta, sub_phi, nside, mostrar = True):
    'Obtiene diversos planos que dividen en dos el mapa de CMB.'
    'Calcula la asimetría entre dos hemisferios de este mapa, para cada plano.'

    # Aplicamos la máscara al mapa (si está definida).
    if mask_gal is not None:
        mapa_cargado = hp.ma(mapa)
        mapa_cargado.mask = np.logical_not(mask_gal)

    else:
        mapa_cargado = mapa        
        
    # Realizamos una rápida comprobación para ahorrarnos tiempo.
    NSIDE = hp.get_nside(mapa)

    if nside != NSIDE:
        mapa_ajustado = hp.ud_grade(mapa_cargado, nside) 

    else:
        mapa_ajustado = mapa_cargado
      
    # Obtenemos los ángulos esféricos (theta: colatitud, phi: longitud) de cada píxel.
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # Lo siguiente es hacer las divisiones de theta y phi para los planos:
    theta_prima = np.linspace(0, np.pi / 2, sub_theta + 1, endpoint=True)[1:]
    phi_prima = np.linspace(0, 2*np.pi, sub_phi, endpoint = False)

    # Generamos ahora una variable que contendrá los vectores normales de los planos
    # Asimismo, añadimos manualmente el plano z = 0 para que no esté evaluado
    # multiples veces.
    planos = [[0.0, 0.0, 1.0]]
    theta_plano = [0]
    phi_plano = [0]

    for i in range(len(phi_prima)):
        for j in range(len(theta_prima)):
        
            x = np.sin(theta_prima[j]) * np.cos(phi_prima[i])
            y = np.sin(theta_prima[j]) * np.sin(phi_prima[i])
            z = np.cos(theta_prima[j])
    
            planos.append([x, y, z])

            # Para que la función nos devuelva los datos de theta y phi pasaremos el coseno de theta
            # a theta usando el arccos, así como phi a grados también. De esta forma, obtendremos
            # directamente los valores de theta y phi para los que la asietría es máxima.
            theta_plano.append(np.degrees(theta_prima[j]))
            phi_plano.append(np.degrees(phi_prima[i]))

    # Generamos entonces el array con estos vectores normales.
    n = np.array(planos)

    # Una vez obtenidos los vectores de los planos, obtenemos los vectores
    # de posición r tal que al hacer el producto escalar de dichos vectores
    # con los del plano obtendremos si estamos situados en el "norte" o el 
    # "sur".
    r = np.column_stack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)])

    # Para evitar usar .compressed() todo el rato, vamos a 
    # seleccionar directamente qué pixeles son válidos tras aplicar
    # la máscara. De esta forma los cálculos son más rápidos.
    if mask_gal is not None:
        mapa_valido = mapa_ajustado.data[np.logical_not(mapa_ajustado.mask)]
        r = r[np.logical_not(mapa_ajustado.mask), :]
    else:
        mapa_valido = mapa_ajustado.data

    sigma_total = np.var(mapa_valido) 

    asimetria = []
    
    # Hacemos el producto escalar plano por plano dentro del bucle
    for j in range(len(n)):
        # Multiplicamos escalarmente todos los puntos de la esfera por el vector
        # del plano en cuestión. De esta forma obtenemos los puntos que pertenecerán
        # al norte, y los que pertenecerán al sur.
        prod_escalar = r @ n[j, :]

        # Los puntos correspondientes al "norte" serán aquellos que apunten en la misma
        # dirección que el vector del plano (r * n > 0).
        norte = (prod_escalar > 0)

        # Los puntos correspondientes al "sur" serán aquellos que apunten dirección
        # contraria al vector del plano (r * n < 0). También podemos verlo como los puntos
        # que no pertenecen al norte.
        sur = ~norte

        # Para cada plano, obtenemos la varianza de su "norte" y de su "sur".
        sigma_norte = np.var(mapa_valido[norte])
        sigma_sur = np.var(mapa_valido[sur])

        A = (sigma_norte - sigma_sur) / sigma_total

        asimetria.append(A)

    # Escogemos finalmente los cinco planos que contienen el máximo de asimetría. Para ello
    # Ordenamos el array de asimetría y tomamos los cinco úlimos elementos (correspondientes
    # a los máximos en valor absoluto)
    i_top5 = np.argsort(np.abs(asimetria))[-5:]

    if mostrar:

        # Si elegimos la opción de mostrar, queremos dibujar en una cuadrícula para cada par phi-theta
        # El plano que estamos considerando. Para ello, obtenemos el rango de filas y columnas teniendo en
        # cuenta que hay que añadir una columna extra debido al plano z = 0.
        columnas = len(phi_prima) + 1         
        filas = len(theta_prima)
        
        fig = plt.figure(figsize=(3.5 * columnas, 2.5 * filas))

        # Para cada theta (fila), obtenemos el mapa recorriendo cada phi (columna).
        for i in range(filas):
            for j in range(columnas):
                
                # Posición del subgráfico en Matplotlib (empieza en 1).
                posicion = i * columnas + j + 1
                
                if j == 0:
                    # La primera columna corresponde al plano z = 0 ([0, 0, 1])
                    # Solo lo pintamos en la primera fila (i=0) para que no se repita abajo.
                    if i == 0:
                        i_plano = 0
                        theta_p = 0.0
                        phi_p = 0.0
                    else:
                        continue
                else:
                    # Si no se trata del primer plano, obtenemos su theta y su phi en grados.
                    i_plano = 1 + (j - 1) * len(theta_prima) + i
                    theta_p = np.degrees(theta_prima[i])
                    phi_p = np.degrees(phi_prima[j - 1])
                
                n_actual = n[i_plano]
                A_actual = asimetria[i_plano]
                
                # Realizamos nuevamente el producto escalar para obtener la posición de los hemisferios
                prod_escalar = r @ n_actual

                # Generamos un mapa completamente vacío, donde si se cumple la condición de que el producto
                # escalar sea mayor que cero, lo rellenaremos de 1. En caso contrario (hemisferio sur) lo 
                # rellenaremos de -1.
                mapa_visual = np.full(npix, np.nan)
                valores_hemisferio = np.where(prod_escalar > 0, 1.0, -1.0)

                if mask_gal is not None:
                    # Si la máscara del plano galáctico ha sido añadida a la función, la vamos a dejar tal cual
                    # De esta forma, los puntos que no pertenezcan a la máscara serán los que pintemos.
                    mapa_visual[np.logical_not(mapa_ajustado.mask)] = valores_hemisferio
                else:
                    # Si no hemos añadido la máscara del plano galáctico, el mapa visual se pinta completamente
                    # con los puntos del hemisferio norte y sur, sin eliminar ninguno.
                    mapa_visual = valores_hemisferio.copy()
    
                    # Y ahora lo transformamos en un mapa enmascarado oficial de Healpy heredando la máscara
                    mapa_visual = hp.ma(mapa_visual)
                    mapa_visual.mask = mapa_ajustado.mask
                    
                # Si el plano está entre los cinco con más asimetría, usamos un color verdoso, si no rojizo.
                mapa_color = "Pastel2" if i_plano in i_top5 else "Pastel1"
                
                # Finalmente pintamos los mapas.
                hp.mollview(mapa_visual, 
                            sub=(filas, columnas, posicion),
                            title=f"θ={theta_p:.0f}°, φ={phi_p:.0f}°\nA={A_actual:.5f}",
                            cmap=mapa_color,        
                            cbar=False,        
                            notext=True)       
                
        plt.tight_layout()
        plt.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - #       
    return np.array(asimetria)[i_top5], np.array(theta_plano)[i_top5], np.array(phi_plano)[i_top5]



def ASPAS(mapa, mask_gal, sub_theta, sub_phi, nside, mostrar = True):
    'Obtiene diversos planos que dividen en dos el mapa de CMB, añadiendo'
    'un plano perpendicular a este dividiendo el cielo en cuatro.'
    'Calcula la asimetría entre todos los cuadrantes de este mapa, para cada plano.'

    # Aplicamos la máscara al mapa (si está definida).
    if mask_gal is not None:
        mapa_cargado = hp.ma(mapa)
        mapa_cargado.mask = np.logical_not(mask_gal)
    else:
        mapa_cargado = mapa    
        
    # Realizamos una rápida comprobación del nside para ahorrarnos tiempo.
    NSIDE = hp.get_nside(mapa)
    if nside != NSIDE:
        mapa_ajustado = hp.ud_grade(mapa_cargado, nside) 
    else:
        mapa_ajustado = mapa_cargado
      
    # Obtenemos los ángulos esféricos (theta: colatitud, phi: longitud) de cada píxel.
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # Lo siguiente es hacer las divisiones de theta y phi para los planos:
    theta_prima = np.linspace(0, np.pi / 2, sub_theta + 1, endpoint=True)[1:]
    phi_prima = np.linspace(0, 2*np.pi, sub_phi, endpoint = False)

    # Generamos ahora una variable que contendrá los vectores normales de los planos
    # Asimismo, añadimos manualmente el plano z = 0 para que no esté evaluado
    # multiples veces, así como el plano y = 0 que divide los hemisferios en dos
    # cuadrantes cada uno.
    planos_hemisferio = [[0.0, 0.0, 1.0]]
    planos_cuadrantes = [[0.0, 1.0, 0.0]]
    theta_plano = [0]
    phi_plano = [0]

    for i in range(len(phi_prima)):
        x_cuad = -1 * np.sin(phi_prima[i])
        y_cuad = 1 * np.cos(phi_prima[i])
        z_cuad = 0
        
        for j in range(len(theta_prima)):
        
            x = np.sin(theta_prima[j]) * np.cos(phi_prima[i])
            y = np.sin(theta_prima[j]) * np.sin(phi_prima[i])
            z = np.cos(theta_prima[j])
    
            planos_hemisferio.append([x, y, z])
            planos_cuadrantes.append([x_cuad, y_cuad, z_cuad])

            # Para que la función nos devuelva los datos de theta y phi pasaremos el coseno de theta
            # a theta usando el arccos, así como phi a grados también. De esta forma, obtendremos
            # directamente los valores de theta y phi para los que la asietría es máxima.
            theta_plano.append(np.degrees(theta_prima[j]))
            phi_plano.append(np.degrees(phi_prima[i]))

    # Generamos entonces el array con estos vectores normales.
    n = np.array(planos_hemisferio)
    m = np.array(planos_cuadrantes)

    # Una vez obtenidos los vectores de los planos, obtenemos los vectores
    # de posición r tal que al hacer el producto escalar de dichos vectores
    # con los del plano obtendremos si estamos situados en el "norte" o el 
    # "sur".
    r = np.column_stack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)])

    # Para evitar usar .compressed() todo el rato, vamos a 
    # seleccionar directamente qué pixeles son válidos tras aplicar
    # la máscara. De esta forma los cálculos son más rápidos.
    if mask_gal is not None:
        mapa_valido = mapa_ajustado.data[np.logical_not(mapa_ajustado.mask)]
        r_valido = r[np.logical_not(mapa_ajustado.mask), :]
    else:
        mapa_valido = mapa_ajustado.data
        r_valido = r

    # Aprovechamos para obtener la varianza del mapa completo.
    sigma_total = np.var(mapa_valido) 

    # Iniciamos los arrays que contendrán las distintas asimetrías.
    asimetria_NS = []
    asimetria_NONE = []
    asimetria_NOSO = []
    asimetria_NOSE = []
    asimetria_NESO = []
    asimetria_NESE = []
    asimetria_SOSE = []
    
    # Preparamos las etiquetas y la lista para almacenar los títulos.
    titulos_asimetrias = ["A_NS", "A_NONE", "A_NOSO", "A_NOSE", "A_NESO", "A_NESE", "A_SOSE"]
    ganadores_titulo = []

    # Hacemos el producto escalar plano por plano dentro del bucle
    for j in range(len(n)):
        # Multiplicamos escalarmente todos los puntos de la esfera por el vector
        # del plano en cuestión. De esta forma obtenemos los puntos que pertenecerán
        # al norte, y los que pertenecerán al sur.
        prod_escalar_hemisferio = r_valido @ n[j, :]
        prod_escalar_cuadrante = r_valido @ m[j, :]

        # Los puntos correspondientes al "norte" serán aquellos que apunten en la misma
        # dirección que el vector del plano (r * n > 0).
        norte = (prod_escalar_hemisferio > 0)
        noreste = (prod_escalar_hemisferio > 0) & (prod_escalar_cuadrante > 0)
        noroeste = (prod_escalar_hemisferio > 0) & (~noreste)

        # Los puntos correspondientes al "sur" serán aquellos que apunten dirección
        # contraria al vector del plano (r * n < 0). También podemos verlo como los puntos
        # que no pertenecen al norte.
        sur = ~norte
        sureste = (~norte) & (prod_escalar_cuadrante > 0)
        suroeste = (~norte) & (~sureste)

        # Para cada plano, obtenemos la varianza de su "norte" y de su "sur".
        sigma_norte = np.var(mapa_valido[norte]) 
        sigma_sur = np.var(mapa_valido[sur]) 
        sigma_noroeste = np.var(mapa_valido[noroeste])
        sigma_noreste = np.var(mapa_valido[noreste])
        sigma_suroeste = np.var(mapa_valido[suroeste])
        sigma_sureste = np.var(mapa_valido[sureste]) 

        A_NS = (sigma_norte - sigma_sur) / sigma_total
        A_NONE = (sigma_noroeste - sigma_noreste) / sigma_total
        A_NOSO = (sigma_noroeste - sigma_suroeste) / sigma_total
        A_NOSE = (sigma_noroeste - sigma_sureste) / sigma_total
        A_NESO = (sigma_noreste - sigma_suroeste) / sigma_total
        A_NESE = (sigma_noreste - sigma_sureste) / sigma_total
        A_SOSE = (sigma_suroeste - sigma_sureste) / sigma_total

        asimetria_NS.append(A_NS)
        asimetria_NONE.append(A_NONE)
        asimetria_NOSO.append(A_NOSO)
        asimetria_NOSE.append(A_NOSE)
        asimetria_NESO.append(A_NESO)
        asimetria_NESE.append(A_NESE)
        asimetria_SOSE.append(A_SOSE)
        
        # Guardamos de forma simple y rápida el nombre y valor de la asimetría máxima en valor absoluto.
        valores = [A_NS, A_NONE, A_NOSO, A_NOSE, A_NESO, A_NESE, A_SOSE]
        idx_max = np.argmax(np.abs(valores))
        ganadores_titulo.append(f"{titulos_asimetrias[idx_max]}={valores[idx_max]:.5f}")

    if mostrar:

        # Si elegimos la opción de mostrar, queremos dibujar en una cuadrícula para cada par phi-theta
        # El plano que estamos considerando. Para ello, obtenemos el rango de filas y columnas teniendo en
        # cuenta que hay que añadir una columna extra debido al plano z = 0.
        columnas = len(phi_prima) + 1         
        filas = len(theta_prima)
        
        fig = plt.figure(figsize=(3.5 * columnas, 2.5 * filas))

        # Modificamos los colores de cada cuadrante
        colores_cuadrantes = [
            "#C0392B",  # Suroeste
            "#F39C12",  # Sureste
            "#BB8FCE",  # Noreste
            "#1A5276"   # Noroeste
        ]
        cmap = ListedColormap(colores_cuadrantes)
        
        # Para cada theta (fila), obtenemos el mapa recorriendo cada phi (columna).
        for i in range(filas):
            for j in range(columnas):
                
                # Posición del subgráfico en Matplotlib (empieza en 1).
                posicion = i * columnas + j + 1
                
                if j == 0:
                    # La primera columna corresponde al plano z = 0 ([0, 0, 1])
                    # Solo lo pintamos en la primera fila (i=0) para que no se repita abajo.
                    if i == 0:
                        i_plano = 0
                        theta_p = 0.0
                        phi_p = 0.0
                    else:
                        continue
                else:
                    # Si no se trata del primer plano, obtenemos su theta y su phi en grados.
                    i_plano = 1 + (j - 1) * len(theta_prima) + i
                    theta_p = np.degrees(theta_prima[i])
                    phi_p = np.degrees(phi_prima[j - 1])
                
                n_actual = n[i_plano]
                m_actual = m[i_plano]
                               
                # Realizamos nuevamente el producto escalar para obtener la posición de los hemisferios.
                # Evaluamos en toda la esfera para poder generar correctamente el mapa visual completo.
                prod_escalar_hemisferio = r @ n_actual
                prod_escalar_cuadrante = r @ m_actual

                # Generamos un mapa completamente vacío, donde asignaremos un valor numérico
                # distinto a cada uno de los cuatro cuadrantes para que se diferencien al graficar.
                mapa_visual = np.full(npix, np.nan)
                
                # Definimos máscaras booleanas para toda la esfera siguiendo tu lógica analítica
                norte_visual = (prod_escalar_hemisferio > 0)
                noreste_visual = (prod_escalar_hemisferio > 0) & (prod_escalar_cuadrante > 0)
                noroeste_visual = (prod_escalar_hemisferio > 0) & (~noreste_visual)
                sureste_visual = (~norte_visual) & (prod_escalar_cuadrante > 0)
                suroeste_visual = (~norte_visual) & (~sureste_visual)
                
                # Asignamos valores fijos discretos (1, 2, 3, 4) a cada cuadrante
                valores_cuadrantes = np.zeros(npix)
                valores_cuadrantes[noreste_visual] = 1.0
                valores_cuadrantes[noroeste_visual] = 2.0
                valores_cuadrantes[suroeste_visual] = 3.0
                valores_cuadrantes[sureste_visual] = 4.0


                # Si la máscara del plano galáctico ha sido añadida a la función, la vamos a dejar tal cual
                # De esta forma, los puntos que no pertenezcan a la máscara serán los que pintemos.
                mapa_visual[np.logical_not(mapa_ajustado.mask)] = valores_cuadrantes[np.logical_not(mapa_ajustado.mask)]
                
                # Lo transformamos en un mapa enmascarado oficial de Healpy heredando la máscara
                mapa_visual = hp.ma(mapa_visual)
                mapa_visual.mask = mapa_ajustado.mask
                
                # Finalmente pintamos los mapas.
                hp.mollview(mapa_visual, 
                            sub=(filas, columnas, posicion),
                            title=f"θ={theta_p:.0f}°, φ={phi_p:.0f}°\n{ganadores_titulo[i_plano]}",
                            cmap=cmap,        
                            cbar=False,        
                            notext=True)       
                plt.gca().title.set_fontsize(23)
                
        plt.tight_layout()
        plt.show()

    # Construimos la matriz con todos los datos recopilados fila por fila.
    phi_segundo_plano = np.array(phi_plano) + 90.0

    theta_redondeado = np.round(theta_plano)
    phi_redondeado = np.round(phi_plano)
    phi_segundo_redondeado = np.round(phi_segundo_plano)

    matriz_valores = np.column_stack([
        phi_redondeado,     
        phi_segundo_redondeado,      
        theta_redondeado,   
        np.array(asimetria_NS),  
        np.array(asimetria_NONE),
        np.array(asimetria_NOSO),
        np.array(asimetria_NOSE),
        np.array(asimetria_NESO),
        np.array(asimetria_NESE),
        np.array(asimetria_SOSE) 
    ])

    # Convertimos a DataFrame.
    df_resultados = pd.DataFrame(matriz_valores)
    
    # Hacemos que las tres primeras columnas sean enteros, ya que son ángulos.
    df_resultados.iloc[:, 0:3] = df_resultados.iloc[:, 0:3].astype(int)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    return df_resultados

    

def ASIMETRIA_SIMULACION(cls, nside, lmax, num_sim, mask_gal_T, mask_gal_P, theta, phi):
    'Obtenemos n simulaciones y calculamos la asimetría respecto a un plano dado'
    'por theta y phi'

    # Lo primero es reducir la máscara tanto como nos indique el nside, ya que originalmente
    # está hecha para nside = 2048.

    NSIDE = hp.get_nside(mask_gal_T)
    if nside != NSIDE:
        mask_gal_ajustada_T = hp.ud_grade(mask_gal_T.astype(float), nside)
        mask_gal_ajustada_T = mask_gal_ajustada_T > 0.99   
    
        mask_gal_ajustada_P = hp.ud_grade(mask_gal_P.astype(float), nside)
        mask_gal_ajustada_P = mask_gal_ajustada_P > 0.99   
    else:
         mask_gal_ajustada_T = mask_gal_T
         mask_gal_ajustada_P = mask_gal_P

    # Guardamos los píxeles válidos para usarlos directamente dentro del bucle.
    pixeles_validos_T = np.logical_not(mask_gal_ajustada_T)
    pixeles_validos_P = np.logical_not(mask_gal_ajustada_P)

    # Obtenemos los ángulos esféricos (theta: colatitud, phi: longitud) de cada píxel.
    npix = hp.nside2npix(nside)
    theta_r, phi_r = hp.pix2ang(nside, np.arange(npix))

    # Obtenemos los vectores de posición r tal que al hacer el producto 
    # escalar de dichos vectores con el del plano obtendremos si estamos 
    # situados en el "norte" o el "sur".
    r = np.column_stack([np.sin(theta_r) * np.cos(phi_r),
                         np.sin(theta_r) * np.sin(phi_r),
                         np.cos(theta_r)])

    # Aplicamos al máscara galáctica ajustada.
    r_T = r[pixeles_validos_T, :]
    r_P = r[pixeles_validos_P, :]
    
    # Obtenemos el vector del plano y regiones fuera del bucle
    x = np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    y = np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    z = np.cos(np.radians(theta))
    n = np.array([x, y, z])

    x_cuad = -1 * np.sin(phi)
    y_cuad = 1 * np.cos(phi)
    z_cuad = 0
    m = np.array([x_cuad, y_cuad, z_cuad])
    
    prod_escalar_hemisferio_T = r_T @ n
    prod_escalar_hemisferio_P = r_P @ n
    prod_escalar_cuadrante_T = r_T @ m
    prod_escalar_cuadrante_P = r_P @ m
    
    norte_T = (prod_escalar_hemisferio_T > 0)
    noreste_T = (prod_escalar_hemisferio_T > 0) & (prod_escalar_cuadrante_T > 0)
    noroeste_T = (prod_escalar_hemisferio_T > 0) & (~noreste_T)

    sur_T = ~norte_T
    sureste_T = (~norte_T) & (prod_escalar_cuadrante_T > 0)
    suroeste_T = (~norte_T) & (~sureste_T)


    norte_P = (prod_escalar_hemisferio_P > 0)
    noreste_P = (prod_escalar_hemisferio_P > 0) & (prod_escalar_cuadrante_P > 0)
    noroeste_P = (prod_escalar_hemisferio_P > 0) & (~noreste_P)

    sur_P = ~norte_P
    sureste_P = (~norte_P) & (prod_escalar_cuadrante_P > 0)
    suroeste_P = (~norte_P) & (~sureste_P)

    # Inicializamos la lista para guardar los resultados de Pandas
    lista_resultados = []

    # Iniciamos la variable de iteración i:
    i = 0
    
    for i in range(num_sim):
        
        # Tomando los cls obtenidos previamente, y el valor del nside obtenemos tantas simulaciones
        # como estén indicadas en num_sim.
        mapa_T, mapa_Q, mapa_U = hp.synfast(cls, nside=nside, lmax=lmax, new=True)

        # En lugar de hp.ma(), extraemos directamente los datos válidos
        # indexando con el array booleano que calculamos al principio. Tus variables mapa_Xxx
        # reciben exactamente el mismo contenido pero de forma ultra rápida.

        mapa_Txx = mapa_T[pixeles_validos_T]
        mapa_Qxx = mapa_Q[pixeles_validos_P]
        mapa_Uxx = mapa_U[pixeles_validos_P]

        # Obtenemos las sigmas totales de cada mapa.
        sigma_total_T = np.var(mapa_Txx)
        sigma_total_Q = np.var(mapa_Qxx)     
        sigma_total_U = np.var(mapa_Uxx)
        
        # Obtenemos finalmente la varianza del norte y del sur para cada
        # mapa T, Q o U.
        sigma_norte_T = np.var(mapa_Txx[norte_T])
        sigma_sur_T = np.var(mapa_Txx[sur_T])

        sigma_norte_T = np.var(mapa_Txx[norte_T]) 
        sigma_sur_T = np.var(mapa_Txx[sur_T]) 
        sigma_noroeste_T = np.var(mapa_Txx[noroeste_T])
        sigma_noreste_T = np.var(mapa_Txx[noreste_T])
        sigma_suroeste_T = np.var(mapa_Txx[suroeste_T])
        sigma_sureste_T = np.var(mapa_Txx[sureste_T])    

        A_NS_T = (sigma_norte_T - sigma_sur_T) / sigma_total_T
        A_NONE_T = (sigma_noroeste_T - sigma_noreste_T) / sigma_total_T
        A_NOSO_T = (sigma_noroeste_T - sigma_suroeste_T) / sigma_total_T
        A_NOSE_T = (sigma_noroeste_T - sigma_sureste_T) / sigma_total_T 
        A_NESO_T = (sigma_noreste_T - sigma_suroeste_T) / sigma_total_T 
        A_NESE_T = (sigma_noreste_T - sigma_sureste_T) / sigma_total_T 
        A_SOSE_T = (sigma_suroeste_T - sigma_sureste_T) / sigma_total_T 


        sigma_norte_Q = np.var(mapa_Qxx[norte_P]) 
        sigma_sur_Q = np.var(mapa_Qxx[sur_P])
        sigma_noroeste_Q = np.var(mapa_Qxx[noroeste_P])
        sigma_noreste_Q = np.var(mapa_Qxx[noreste_P])
        sigma_suroeste_Q = np.var(mapa_Qxx[suroeste_P])
        sigma_sureste_Q = np.var(mapa_Qxx[sureste_P])

        A_NS_Q = (sigma_norte_Q - sigma_sur_Q) / sigma_total_Q
        A_NONE_Q = (sigma_noroeste_Q - sigma_noreste_Q) / sigma_total_Q
        A_NOSO_Q = (sigma_noroeste_Q - sigma_suroeste_Q) / sigma_total_Q 
        A_NOSE_Q = (sigma_noroeste_Q - sigma_sureste_Q) / sigma_total_Q 
        A_NESO_Q = (sigma_noreste_Q - sigma_suroeste_Q) / sigma_total_Q 
        A_NESE_Q = (sigma_noreste_Q - sigma_sureste_Q) / sigma_total_Q 
        A_SOSE_Q = (sigma_suroeste_Q - sigma_sureste_Q) / sigma_total_Q 


        sigma_norte_U = np.var(mapa_Uxx[norte_P])
        sigma_sur_U = np.var(mapa_Uxx[sur_P])
        sigma_noroeste_U = np.var(mapa_Uxx[noroeste_P])
        sigma_noreste_U = np.var(mapa_Uxx[noreste_P])
        sigma_suroeste_U = np.var(mapa_Uxx[suroeste_P])
        sigma_sureste_U = np.var(mapa_Uxx[sureste_P])       

        A_NS_U = (sigma_norte_U - sigma_sur_U) / sigma_total_U
        A_NONE_U = (sigma_noroeste_U - sigma_noreste_U) / sigma_total_U
        A_NOSO_U = (sigma_noroeste_U - sigma_suroeste_U) / sigma_total_U 
        A_NOSE_U = (sigma_noroeste_U - sigma_sureste_U) / sigma_total_U 
        A_NESO_U = (sigma_noreste_U - sigma_suroeste_U) / sigma_total_U 
        A_NESE_U = (sigma_noreste_U - sigma_sureste_U) / sigma_total_U 
        A_SOSE_U = (sigma_suroeste_U - sigma_sureste_U) / sigma_total_U


        datos_simulacion = {
            # Temperatura (T)
            'A_NS_T': A_NS_T, 'A_NONE_T': A_NONE_T, 'A_NOSO_T': A_NOSO_T, 'A_NOSE_T': A_NOSE_T, 
            'A_NESO_T': A_NESO_T, 'A_NESE_T': A_NESE_T, 'A_SOSE_T': A_SOSE_T,
            
            # Polarización (Q)
            'A_NS_Q': A_NS_Q, 'A_NONE_Q': A_NONE_Q, 'A_NOSO_Q': A_NOSO_Q, 'A_NOSE_Q': A_NOSE_Q, 
            'A_NESO_Q': A_NESO_Q, 'A_NESE_Q': A_NESE_Q, 'A_SOSE_Q': A_SOSE_Q,
            
            # Polarización (U)
            'A_NS_U': A_NS_U, 'A_NONE_U': A_NONE_U, 'A_NOSO_U': A_NOSO_U, 'A_NOSE_U': A_NOSE_U, 
            'A_NESO_U': A_NESO_U, 'A_NESE_U': A_NESE_U, 'A_SOSE_U': A_SOSE_U
        }
        
        lista_resultados.append(datos_simulacion)
        
    df_asimetrias = pd.DataFrame(lista_resultados)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    return df_asimetrias



def ASIMETRIA_SIMULACION_NS(cls, nside, lmax, num_sim, mask_gal_T, mask_gal_P, theta, phi):
    'Obtenemos n simulaciones y calculamos la asimetría respecto a un plano dado'
    'por theta y phi'

    # Lo primero es reducir la máscara tanto como nos indique el nside, ya que originalmente
    # está hecha para nside = 2048.
    mask_gal_ajustada_T = hp.ud_grade(mask_gal_T.astype(float), nside)
    mask_gal_ajustada_T = mask_gal_ajustada_T > 0.99   

    mask_gal_ajustada_P = hp.ud_grade(mask_gal_P.astype(float), nside)
    mask_gal_ajustada_P = mask_gal_ajustada_P > 0.99   

    # Guardamos los píxeles válidos para usarlos directamente dentro del bucle.
    pixeles_validos_T = np.logical_not(mask_gal_ajustada_T)
    pixeles_validos_P = np.logical_not(mask_gal_ajustada_P)

    # Obtenemos los ángulos esféricos (theta: colatitud, phi: longitud) de cada píxel.
    npix = hp.nside2npix(nside)
    theta_r, phi_r = hp.pix2ang(nside, np.arange(npix))

    # Obtenemos los vectores de posición r tal que al hacer el producto 
    # escalar de dichos vectores con el del plano obtendremos si estamos 
    # situados en el "norte" o el "sur".
    r = np.column_stack([np.sin(theta_r) * np.cos(phi_r),
                         np.sin(theta_r) * np.sin(phi_r),
                         np.cos(theta_r)])

    # Aplicamos al máscara galáctica ajustada.
    r = r[pixeles_validos, :]
    
    # [CAMBIO ANTERIOR] Obtenemos el vector del plano y regiones fuera del bucle
    x = np.sin(np.radians(theta)) * np.cos(np.radians(phi))
    y = np.sin(np.radians(theta)) * np.sin(np.radians(phi))
    z = np.cos(np.radians(theta))
    n = np.array([x, y, z])
    
    prod_escalar = r @ n
    norte = (prod_escalar > 0)    
    sur = ~norte   

    # Iniciamos la variable de iteración i:
    i = 0

    # Generamos un array donde iremos guardando la asimetría de cada mapa.
    asimetria = []
    
    for i in range(num_sim):
        
        # Tomando los cls obtenidos previamente, y el valor del nside obtenemos tantas simulaciones
        # como estén indicadas en num_sim.
        mapa_T, mapa_Q, mapa_U = hp.synfast(cls, nside=nside, lmax=lmax, new=True)

        # [ACELERACIÓN MÁSCARAS]: En lugar de hp.ma(), extraemos directamente los datos válidos
        # indexando con el array booleano que calculamos al principio. Tus variables mapa_Xxx
        # reciben exactamente el mismo contenido pero de forma ultra rápida.
        mapa_Txx = mapa_T[pixeles_validos]
        mapa_Qxx = mapa_Q[pixeles_validos]
        mapa_Uxx = mapa_U[pixeles_validos]

        # Obtenemos las sigmas totales de cada mapa.
        sigma_total_T = np.var(mapa_Txx)
        sigma_total_Q = np.var(mapa_Qxx)     
        sigma_total_U = np.var(mapa_Uxx)
        
        # Obtenemos finalmente la varianza del norte y del sur para cada
        # mapa T, Q o U.
        sigma_norte_T = np.var(mapa_Txx[norte])
        sigma_sur_T = np.var(mapa_Txx[sur])
        asimetria_T = (sigma_norte_T - sigma_sur_T) / sigma_total_T   

        sigma_norte_Q = np.var(mapa_Qxx[norte])
        sigma_sur_Q = np.var(mapa_Qxx[sur])
        asimetria_Q = (sigma_norte_Q - sigma_sur_Q) / sigma_total_Q

        sigma_norte_U = np.var(mapa_Uxx[norte])
        sigma_sur_U = np.var(mapa_Uxx[sur])
        asimetria_U = (sigma_norte_U - sigma_sur_U) / sigma_total_U 

        # Anexamos estas asimetrías al array asimetría.
        asimetria.append([asimetria_T, asimetria_Q, asimetria_U])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    return np.array(asimetria)