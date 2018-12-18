import numpy as np 

import math

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

from matplotlib.patches import Ellipse

iteration = 0

def plotgraphs(X,mean1,covariance1,mean2,covariance2,mean3,covariance3,ux):
    x = []
    y = []
    for i in range(len(X)):
        x.append(X.item((i,0)))
        y.append(X.item((i,1)))
    plt.scatter(x,y,marker = '^')
    plot_cov_ellipse(covariance1,mean1,nstd=2,ax=None,alpha = 0.5,color= 'r')
    plot_cov_ellipse(covariance2,mean2,nstd=2,ax=None,alpha = 0.5,color= 'g')
    plot_cov_ellipse(covariance3,mean3,nstd=2,ax=None,alpha = 0.5,color= 'b')

    string = 'task3_gmm_iter' + str(ux) + '.jpg'
    plt.savefig(string)
    plt.close()


    return


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def calculateGamma(X,phi,summation,mean ,covariance):
    
    gamma = (phi * multivariate_normal.pdf(X,mean= mean, cov= covariance) ) / summation
    print(gamma.shape)


    return gamma

def calculateGMM(X,mean1,mean2,mean3,phi,covariance1,covariance2,covariance3,ux):

    global iteration
    print(iteration,ux)

    if ux == 7:
        return

    summation = phi.item(0) * multivariate_normal.pdf(X, mean= mean1, cov= covariance1) + phi.item(1) * multivariate_normal.pdf(X, mean= mean2, cov= covariance2) + phi.item(2) * multivariate_normal.pdf(X, mean= mean3, cov= covariance3)

    #print(summation.shape)

    #print(summation)

    gamma1 = calculateGamma(X,phi.item(0),summation,mean1,covariance1)
    
    gamma2 = calculateGamma(X,phi.item(1),summation,mean2,covariance2)
    
    gamma3 = calculateGamma(X,phi.item(2),summation,mean3,covariance3)

    print(gamma1)

    print(gamma2)

    print(gamma3)

    new_mean_1 = np.zeros((1,mean1.shape[0]))
    
    new_mean_2 = np.zeros((1,mean2.shape[0]))

    new_mean_3 = np.zeros((1,mean2.shape[0]))

    for i in range(len(X)):

        new_mean_1 += gamma1[i] * X[i]

        new_mean_2 += gamma2[i] * X[i]

        new_mean_3 += gamma3[i] * X[i]

    new_mean_1 = new_mean_1/ np.sum(gamma1)

    new_mean_2 = new_mean_2/ np.sum(gamma2)

    new_mean_3 = new_mean_3/ np.sum(gamma3)

    new_mean_1 = new_mean_1.reshape(new_mean_1.shape[1])

    new_mean_2 = new_mean_2.reshape(new_mean_2.shape[1])

    new_mean_3 = new_mean_3.reshape(new_mean_3.shape[1])

    print(new_mean_1)

    print(new_mean_2)

    print(new_mean_3)

    new_covariance_1 = np.zeros((covariance1.shape))

    new_covariance_2 = np.zeros((covariance2.shape))

    new_covariance_3 = np.zeros((covariance3.shape))

    for i in range(len(X)):

        difference_1 = X[i] - new_mean_1

        difference_2 = X[i] - new_mean_2

        difference_3 = X[i] - new_mean_3

        new_covariance_1 += gamma1[i] * np.dot(difference_1.T,difference_1)

        new_covariance_2 += gamma2[i] * np.dot(difference_2.T,difference_2)

        new_covariance_3 += gamma3[i] * np.dot(difference_3.T, difference_3)

    new_covariance_1 = new_covariance_1/ np.sum(gamma1)

    new_covariance_2 = new_covariance_2/ np.sum(gamma2)

    new_covariance_3 = new_covariance_3/ np.sum(gamma3)

    print(new_covariance_1)

    print(new_covariance_2)

    print(new_covariance_3)

    new_phi = np.zeros((phi.shape))

    new_phi[0][0] = np.sum(gamma1) / X.shape[0]

    new_phi[0][1] = np.sum(gamma2) / X.shape[0]

    new_phi[0][2] = np.sum(gamma3) / X.shape[0]

    print(new_phi)


    plotgraphs(X,mean1,covariance1,mean2,covariance2,mean3,covariance3, iteration)

    iteration = iteration + 1

    calculateGMM(X,new_mean_1,new_mean_2,new_mean_3,new_phi,new_covariance_1,new_covariance_2,new_covariance_3, iteration)


if __name__ == "__main__":

    X = np.matrix('5.9, 3.2;4.6, 2.9;6.2, 2.8;4.7, 3.2;5.5, 4.2;5.0, 3.0;4.9, 3.1;6.7, 3.1;5.1, 3.8;6.0, 3.0')
    
    mean1 = np.array([6.2, 3.2])
    
    mean2 = np.array([6.6, 3.7])
    
    mean3 = np.array([6.5, 3.0])

    phi = np.matrix('0.3333,0.3333,0.3333')

    covariance1 = np.matrix('[0.5, 0 ]; [0 , 0.5]')
    
    covariance2 = np.matrix('[0.5, 0 ]; [0 , 0.5]')
    
    covariance3 = np.matrix('[0.5, 0 ]; [0 , 0.5]')
    
    #calculateGMM(X,mean1,mean2,mean3,phi,covariance1,covariance2,covariance3,0)
    
    X = np.matrix([[3.600,79],[1.800,54],[3.333,74],[2.283,62],[4.533,85],[2.883,55],[4.700,88],[3.600,85],[1.950,51],[4.350,85],[1.833,54],[3.917,84],[4.200,78],[1.750,47],[4.700,83],[2.167,52],[1.750,62],[4.800,84],[1.600,52],[4.250,79],[1.800,51],[1.750,47],[3.450,78],[3.067,69],[4.533,74],[3.600,83],[1.967,55],[4.083,76],[3.850,78],[4.433,79],[4.300,73],[4.467,77],[3.367,66],[4.033,80],[3.833,74],[2.017,52],[1.867,48],[4.833,80],[1.833,59],[4.783,90],[4.350,80],[1.883,58],[4.567,84],[1.750,58],[4.533,73],[3.317,83],[3.833,64],[2.100,53],[4.633,82],[2.000,59],[4.800,75],[4.716,90],[1.833,54],[4.833,80],[1.733,54],[4.883,83],[3.717,71],[1.667,64],[4.567,77],[4.317,81],[2.233,59],[4.500,84],[1.750,48],[4.800,82],[1.817,60],[4.400,92],[4.167,78],[4.700,78],[2.067,65],[4.700,73],[4.033,82],[1.967,56],[4.500,79],[4.000,71],[1.983,62],[5.067,76],[2.017,60],[4.567,78],[3.883,76],[3.600,83],[4.133,75],[4.333,82],[4.100,70],[2.633,65],[4.067,73],[4.933,88],[3.950,76],[4.517,80],[2.167,48],[4.000,86],[2.200,60],[4.333,90],[1.867,50],[4.817,78],[1.833,63],[4.300,72],[4.667,84],[3.750,75],[1.867,51],[4.900,82],[2.483,62],[4.367,88],[2.100,49],[4.500,83],[4.050,81],[1.867,47],[4.700,84],[1.783,52],[4.850,86],[3.683,81],[4.733,75],[2.300,59],[4.900,89],[4.417,79],[1.700,59],[4.633,81],[2.317,50],[4.600,85],[1.817,59],[4.417,87],[2.617,53],[4.067,69],[4.250,77],[1.967,56],[4.600,88],[3.767,81],[1.917,45],[4.500,82],[2.267,55],[4.650,90],[1.867,45],[4.167,83],[2.800,56],[4.333,89],[1.833,46],[4.383,82],[1.883,51],[4.933,86],[2.033,53],[3.733,79],[4.233,81],[2.233,60],[4.533,82],[4.817,77],[4.333,76],[1.983,59],[4.633,80],[2.017,49],[5.100,96],[1.800,53],[5.033,77],[4.000,77],[2.400,65],[4.600,81],[3.567,71],[4.000,70],[4.500,81],[4.083,93],[1.800,53],[3.967,89],[2.200,45],[4.150,86],[2.000,58],[3.833,78],[3.500,66],[4.583,76],[2.367,63],[5.000,88],[1.933,52],[4.617,93],[1.917,49],[2.083,57],[4.583,77],[3.333,68],[4.167,81],[4.333,81],[4.500,73],[2.417,50],[4.000,85],[4.167,74],[1.883,55],[4.583,77],[4.250,83],[3.767,83],[2.033,51],[4.433,78],[4.083,84],[1.833,46],[4.417,83],[2.183,55],[4.800,81],[1.833,57],[4.800,76],[4.100,84],[3.966,77],[4.233,81],[3.500,87],[4.366,77],[2.250,51],[4.667,78],[2.100,60],[4.350,82],[4.133,91],[1.867,53],[4.600,78],[1.783,46],[4.367,77],[3.850,84],[1.933,49],[4.500,83],[2.383,71],[4.700,80],[1.867,49],[3.833,75],[3.417,64],[4.233,76],[2.400,53],[4.800,94],[2.000,55],[4.150,76],[1.867,50],[4.267,82],[1.750,54],[4.483,75],[4.000,78],[4.117,79],[4.083,78],[4.267,78],[3.917,70],[4.550,79],[4.083,70],[2.417,54],[4.183,86],[2.217,50],[4.450,90],[1.883,54],[1.850,54],[4.283,77],[3.950,79],[2.333,64],[4.150,75],[2.350,47],[4.933,86],[2.900,63],[4.583,85],[3.833,82],[2.083,57],[4.367,82],[2.133,67],[4.350,74],[2.200,54],[4.450,83],[3.567,73],[4.500,73],[4.150,88],[3.817,80],[3.917,71],[4.450,83],[2.000,56],[4.283,79],[4.767,78],[4.533,84],[1.850,58],[4.250,83],[1.983,43],[2.250,60],[4.750,75],[4.117,81],[2.150,46],[4.417,90],[1.817,46],[4.467,74]])

    mean1 = np.array([4.0, 81])
    
    mean2 = np.array([2.0, 57])
    
    mean3 = np.array([4.0, 71])

    phi = np.matrix('0.3333,0.3333,0.3333')
    
    covariance1 = np.matrix('[1.30, 13.98 ]; [13.98 , 184.82]')
    
    covariance2 = np.matrix('[1.30, 13.98]; [13.98 , 184.82]')
    
    covariance3 = np.matrix('[1.30, 13.98]; [13.98 , 184.82]')

    i = 0

    calculateGMM(X,mean1,mean2,mean3,phi,covariance1,covariance2,covariance3,0)

