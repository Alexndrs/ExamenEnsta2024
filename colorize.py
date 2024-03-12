import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from PIL import Image 
import numpy as np
from numpy import linalg
import scipy as sp
from scipy import sparse
import time

gray_img = "example.bmp"
marked_img = "example_marked.bmp"
output = "example.png"

niters = 3000
epsilon = 1.E-10

HUE       = 0
SATURATION= 1
INTENSITY = 2

Y         = 0
CB        = 1
CR        = 2

def create_field( values, ifield, nb_layers, prolong_field = False ):
    """
    A partir d'un tableau de taille ny x nx x nfields, on extrait
    le ifield eme champs qu'on stocke dans un nouveau tableau auquel on rajoute
    nb_layers couches de cellules fantomes. Par defaut, ces cellules fantomes
    seront initialisees avec des valeurs nulles, mais si prolong_field est vrai,
    les cellules fantomes sont initialisees en prolongeant les valeurs au bord
    de l'image.
    """
    assert(ifield >= 0)
    assert(ifield < values.shape[-1])
    # A partir d'un tableau contenant n champs de taille ny x nx, on 
    field = np.zeros((values.shape[0]+2*nb_layers,values.shape[1]+2*nb_layers),dtype=np.double)
    # On copie l'intensite dans les pixels non fantome. On normalise ces valeurs entre 0 et 1 :
    field[nb_layers:-nb_layers,nb_layers:-nb_layers] = values[:,:,ifield]
    # Par defaut, les valeurs qu'on prend en condition limite dans les ghosts cells sera zero
    # Si prolong_field est vrai, on prolonge les valeurs aux bords de l'image dans les conditions limites :
    if prolong_field:
        for ilayer in range(nb_layers):
            field[ilayer,nb_layers:-nb_layers] = field[nb_layers,nb_layers:-nb_layers]
            field[-ilayer-1,nb_layers:-nb_layers] = field[-nb_layers-1,nb_layers:-nb_layers]
        for ilayer in range(nb_layers):
            field[:,ilayer] = field[:,nb_layers]
            field[:,-ilayer-1] = field[:,-nb_layers-1]
    return field


def compute_means(intensity):
    """
    Calcule la moyenne de l'intensite d'un pixel et de ses voisins immediats (diagonale comprise)
    en utilisant la couche la plus extérieure des cellules fantomes de l'intensite comme "condition limite"
    """
    return np.array([[(1./9.)*np.sum(intensity[i-1:i+2,j-1:j+2])
                      for j in range(1,intensity.shape[1]-1)] for i in range(1,intensity.shape[0]-1)])


def compute_variance(intensity, means):
    """
    Calcule la variance de l'intensite pour chaque pixel et de ses voisins immediats en utilisant la moyenne
    et l'intensite dont on utilise la couche la plus extérieure des cellules fantomes de l'intensite comme
    "condition limite".
    """
    return np.array([[np.sum(np.power(intensity[i-1:i+2,j-1:j+2]-means[i-1,j-1],2))
                      for j in range(1,intensity.shape[1]-1)] for i in range(1,intensity.shape[0]-1)])


def compute_wrs(intensity, means, variance, ir, jr, ic, jc):
    """
    Calcule un poids pour la contribution d'un pixel voisin (ic,jc) au pixel de coordonne (ir,jr)
    en fonction de la correlation de l'intensite du pixel (ic,jc) avec le pixel (ir,jr)
    """
    # Prise en compte de la variance quand elle est nulle
    sigma = max(variance[ir,jr],0.000002)
    mu_r  = means[ir,jr]
    # +1 sur les index pour intensity a cause de la couche supplementaire de ghost cell pour intensity
    return 1.+(intensity[ir+1,jr+1]-mu_r)*(intensity[ic+1,jc+1]-mu_r)/sigma

def assembly_row( image_size, i_start, intensity, means, variance, pos, pt_rows, ind_cols, coefs):
    """
    Assemble la ligne de la matrice correspondant au pixel se trouvant a la position pos = (i,j)

    Le stockage de la matrice est un stockage morse, c'est à dire :

    Entrees :

    image_size represente la taille de l'image complete (sans cellules fantomes)
    intensity  L'intensite pour chaque pixel, avec deux couches de cellules fantomes
    means      La moyenne de chaque pixel avec ses voisins avec une couche de cellules fantomes
    variance   La variance de chaque pixel avec ses voisins avec une couche de cellules fantomes

    Sorties:
    
    pt_rows represente le debut de chaque ligne de la matrice dans les tableaux ind_cols et coefs
    ind_cols represente les indices colonnes de chaque element non nul de la matrice
    coefs stocke les coefficients non nuls de la matrice
    """
    # nnz : nombre coefficients non nuls sur la ligne de la matrice :
    nnz = 9
    i_glob = i_start + pos[0]
    if   ( i_glob == 0 or i_glob == image_size[1]-1 ) and ( pos[1] == 0 or pos[1] == image_size[0]-1):
        nnz = 4 
    elif ( i_glob == 0 or i_glob == image_size[1]-1 ) and not ( pos[1] == 0 or pos[1] == image_size[0]-1):
        nnz = 6
    elif not ( i_glob == 0 or i_glob == image_size[1]-1 ) and ( pos[1] == 0 or pos[1] == image_size[0]-1):
        nnz = 6
    
    index = pos[0]*image_size[0]+pos[1]
    # Calcul de la position de la ligne suivante dans la matrice :
    pt_rows[index+1] = pt_rows[index] + nnz
    # On commence a remplir ind_cols et coefs avec les coefficinets adequats pour la matrice
    start = pt_rows[index]
    nx = image_size[0]
    ny = image_size[1]
    sum = 0.
    if i_glob >0:
        if pos[1] > 0:
            ind_cols[start] = index - nx - 1
            wrs = compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]-1,pos[1]-1)
            sum += wrs
            coefs[start] = -wrs
            start += 1
        ind_cols[start] = index - nx
        wrs = compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]-1,pos[1])
        sum += wrs
        coefs[start] = -wrs
        start += 1
        if pos[1] < nx-1:
            ind_cols[start] = index - nx + 1
            wrs = compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]-1,pos[1]+1)
            sum += wrs
            coefs[start] = -wrs
            start += 1
    if pos[1] > 0:
        ind_cols[start] = index - 1
        wrs = compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0],pos[1]-1)
        sum += wrs
        coefs[start] = -wrs
        start += 1
    pos_diag = start
    ind_cols[start] = index
    coefs[start] = +1.
    start += 1
    if pos[1] < nx-1:
        ind_cols[start] = index + 1
        wrs = compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0],pos[1]+1)
        sum += wrs
        coefs[start] = -wrs
        start += 1
    if i_glob < ny-1:
        if pos[1] > 0:
            ind_cols[start] = index + nx - 1
            wrs = compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]+1,pos[1]-1)
            sum += wrs
            coefs[start] = -wrs
            start += 1
        ind_cols[start] = index + nx
        wrs = compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]+1,pos[1])
        sum += wrs
        coefs[start] = -wrs
        start += 1
        if pos[1] < nx-1:
            ind_cols[start] = index + nx + 1
            wrs = compute_wrs(intensity, means, variance, pos[0],pos[1],pos[0]+1,pos[1]+1)
            sum += wrs
            coefs[start] = -wrs
            start += 1
    # Normalisation des coefficients
    start0 = pt_rows[index]
    coefs[start0:pos_diag] /= sum
    coefs[pos_diag+1:start] /= sum

def compute_matrix(image_size, i_start, intensity, means, variance):
    """
    Calcule la matrice issue de la minimisation de la fonction quadratique
    """
    ny = means.shape[0] - 2
    nx = means.shape[1] - 2
    # Dimension de la matrice (rectangulaire pour interaction avec ghost cells )
    dim  = nx*ny
    # Nombre d'elements non nuls prevus pour la matrice :
    nnz = 9*(nx-2)*(ny-2) + 12*((nx-2)+(ny-2)) + 16
    if i_start > 0:
        nnz += 3*(nx-2) - 8
    if i_start + ny < image_size[1]-1:
        nnz += 3*(nx-2) - 8
    # Indices du début des lignes dans les tableaux indCols et coefficients :
    beg_rows = np.zeros(dim+1, dtype=np.int64)
    # Indices colonnes des elements non nuls :
    ind_cols = np.empty(nnz, dtype=np.int64)
    coefs    = np.empty(nnz, dtype=np.double)

    # Pour chaque pixel (irow, icol); on fait correspondre la ligne de la matrice d'indice irow*nx + icol
    # On assemble la matrice ligne par ligne
    for irow in range(ny):
        for jcol in range(nx):
            assembly_row(image_size, i_start, intensity, means, variance, (irow,jcol), beg_rows, ind_cols, coefs)
    assert(beg_rows[-1] == nnz)
    # On retourne la matrice sous forme d'une matrice creuse stockee en csr avec scipy
    return sparse.csr_matrix((coefs, ind_cols, beg_rows), dtype=np.double)

def search_fixed_colored_pixels(mark_values):
    """
    Recherche dans l'image marquee l'indice des pixels dont on a fixé la couleur:
    On utilise pour cela l'espace colorimetrique HSV qui separe bien l'intensite
    de la saturation et de la teinte pour chaque pixel :
    """
    hue        = np.array(mark_values[:,:,HUE].flat, dtype=np.double)
    saturation = np.array(mark_values[:,:,SATURATION].flat, dtype=np.double)
    return np.nonzero((hue != 0.) * (saturation != 0.))[0]

def apply_dirichlet(A : sparse.csr_matrix, dirichlet : np.array):
    """
    Applique une condition de dirichlet aux endroits ou la couleur est deja definie a l'initiation
    """
    for irow in range(A.shape[0]):
        if irow in dirichlet:
            A.data[A.indptr[irow]:A.indptr[irow+1]] = [0. if A.indices[i]!=irow else 1. for i in range(A.indptr[irow],A.indptr[irow+1])]
        else:
            for jcol in range(A.indptr[irow],A.indptr[irow+1]):
                if A.indices[jcol] in dirichlet:
                    A.data[jcol] = 0.


def minimize( A : sparse.csr_matrix, b : np.array, x0 : np.array, niters : int, epsilon : float):
    """
    Minimise la fonction quadratique a l'aide d'un gradient conjugue
    """
    r = b-A.dot(x0)
    nrm_r0 = linalg.norm(r)
    gc = A.transpose().dot(r)
    x = np.copy(x0)
    p = np.copy(gc)
    cp = A.dot(p)
    nrm_gc = linalg.norm(gc)
    nrm_cp = linalg.norm(cp)
    alpha = nrm_gc*nrm_gc/(nrm_cp*nrm_cp)
    x += alpha*p
    r -= alpha*cp
    nrm_r = linalg.norm(r)
    gp = np.copy(gc)
    nrm_gp = nrm_gc
    gc = A.transpose().dot(r)
    for i in range(1,niters):
        print(f"Iteration {i:06}/{niters:06} -> ||r||/||r0|| = {nrm_r/nrm_r0:16.14}",end='\r')
        nrm_gc = linalg.norm(gc)
        if nrm_gc < 1.E-14: return x
        beta = -nrm_gc*nrm_gc/(nrm_gp*nrm_gp)
        p = gc - beta*p
        cp = A.dot(p)
        nrm_cp = linalg.norm(cp)
        alpha = nrm_gc*nrm_gc/(nrm_cp*nrm_cp)
        x += alpha*p
        r -= alpha*cp
        gp = np.copy(gc)
        nrm_gp = nrm_gc
        gc = A.transpose().dot(r)
        nrm_r = linalg.norm(r)
        if nrm_r < epsilon*nrm_r0: break 
    return x


if __name__ == '__main__':
    # On va charger l'image afin de lire l'intensite de chaque pixel.
    # Puis on va creer un tableau contenant deux couches de cellules fantomes
    # pour pouvoir calculer facilement la moyenne puis la variance de chaque pixel
    # avec ses huit voisins immediats.
    im_gray = Image.open(gray_img)
    im_gray = im_gray.convert('HSV')
    # On convertit l'image en tableau (ny x nx x 3) (Trois pour les trois composantes de la couleur)
    values_gray = np.array(im_gray)
    # On créer le tableau d'intensite en rajoutant deux couches de cellules fantomes dans chaque direction :
    intensity = (1./255.)*create_field(values_gray, INTENSITY, nb_layers=2, prolong_field=True)

    # Calcul de la moyenne de l'intensite pour chaque pixel avec ses huit voisins
    # La moyenne contient une couche de cellules fantomes (une de moins que l'intensite)
    deb = time.time()
    means = compute_means(intensity)
    end = time.time() - deb
    print(f"Temps calcul moyenne : {end} secondes")
    # Calcul de la variance de l'intensite pour chaque pixel avec ses huit voisins
    # La variance contient une couche de cellules fantomes comme la moyenne.
    deb = time.time()
    variance = compute_variance(intensity, means)
    end = time.time() - deb
    print(f"Temps calcul variance : {end} secondes")

    # Calcul de la matrice utilisee pour minimiser la fonction quadratique
    deb = time.time()
    A = compute_matrix((means.shape[1]-2,means.shape[0]-2), 0, intensity, means, variance)
    end = time.time() - deb
    print(f"Temps calcul matrice : {end} secondes")

    # Calcul des seconds membres
    im = Image.open(marked_img)
    im_ycbcr = im.convert('YCbCr')
    val_ycbcr = np.array(im_ycbcr)
    # Les composantes Cb (bleu) et Cr (Rouge) sont normalisees :
    Cb = (1./255.)*np.array(val_ycbcr[:,:,CB].flat, dtype=np.double)
    Cr = (1./255.)*np.array(val_ycbcr[:,:,CR].flat, dtype=np.double)

    deb=time.time()
    b_Cb = -A.dot(Cb)
    b_Cr = -A.dot(Cr)
    end = time.time() - deb
    print(f"Temps calcul des deux seconds membres : {end} secondes")

    im_hsv = im.convert("HSV")
    val_hsv = np.array(im_hsv)
    deb = time.time()
    fix_coul_indices = search_fixed_colored_pixels(val_hsv)
    end = time.time() - deb
    print(f"Temps recherche couleur fixee : {end} secondes")

    # Application de la condition de Dirichlet sur la matrice :    
    deb = time.time()
    apply_dirichlet(A, fix_coul_indices)
    end = time.time() - deb
    print(f"Temps application dirichlet sur matrice : {end} secondes")

    print(f"Minimisation de la quadratique pour la composante Cb de l'image couleur")
    deb=time.time()
    x0 = np.zeros(Cb.shape,dtype=np.double)
    new_Cb = Cb + minimize(A, b_Cb, x0, niters,epsilon)
    print(f"\nTemps calcul min Cb : {time.time()-deb}")

    print(f"Minimisation de la quadratique pour la composante Cr de l'image couleur")
    deb=time.time()
    x0 = np.zeros(Cr.shape,dtype=np.double)
    new_Cr = Cr + minimize(A, b_Cr, x0, niters,epsilon)
    print(f"\nTemps calcul min Cr : {time.time()-deb}")

    # On remet les valeurs des trois composantes de l'image couleur YCbCr entre 0 et 255 :
    new_Cb *= 255.
    new_Cr *= 255.
    intensity *= 255.

    # Puis on sauve l'image dans un fichier :
    shape = (means.shape[0]-2,means.shape[1]-2)
    new_image_array = np.empty((shape[0],shape[1],3), dtype=np.uint8)
    new_image_array[:,:,0] = intensity[2:-2,2:-2].astype('uint8')
    new_image_array[:,:,1] = np.reshape(new_Cb, shape).astype('uint8')
    new_image_array[:,:,2] = np.reshape(new_Cr, shape).astype('uint8')
    new_im = Image.fromarray(new_image_array, mode='YCbCr')
    new_im.convert('RGB').save(output, 'PNG')
