import sncosmo
from astropy.cosmology import FlatLambdaCDM
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util.data_util import cps2magnitude
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
from datetime import date
from lenstronomy.SimulationAPI.observation_api import SingleBand

from scipy.stats import norm
import random

from scipy.stats import skewnorm

class AsymmetricGaussian():

    def __init__(self, mu, sigma_minus, sigma_plus, discretization, x_range):
        self.x_space = np.linspace(*x_range, num=discretization)
        self.factor = self.rescale_factor(mu=mu, sigma_minus=sigma_minus, sigma_plus=sigma_plus)
        x_weights = np.asarray([self.pdf(x=i, mu=mu, sigma_minus=sigma_minus, sigma_plus=sigma_plus) for i in self.x_space])
        self.norm = 1/np.sum(x_weights)
        self.x_weights = x_weights*self.norm

    def rescale_factor(self, mu, sigma_minus, sigma_plus):
        left_side = norm.pdf(mu, loc=mu, scale=sigma_minus)
        right_side = norm.pdf(mu, loc=mu, scale=sigma_plus)
        return left_side/right_side
        
    def pdf(self, x, mu, sigma_minus, sigma_plus):
        if x <= mu:
            return norm.pdf(x, loc=mu, scale=sigma_minus)
        else:
            return norm.pdf(x, loc=mu, scale=sigma_plus)*self.factor
        
    def sample(self):
        return np.random.choice(self.x_space, p=self.x_weights)
    
class NormalDist():

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self):
        return np.random.normal(loc=self.mu, scale=self.sigma, size=None)

class MagnitudeDist():
    def __init__(self, alpha, beta, x1, c, m_ref):
        self.alpha = alpha
        self.beta = beta
        self.x1 = x1
        self.c = c
        self.m_ref = m_ref

    def value(self, a):
        try:
            return float(a)
        except TypeError:
            return a()

    def sample(self):
        return -1*self.value(self.alpha)*self.value(self.x1) + self.value(self.beta)*self.value(self.c) + self.value(self.m_ref)
    
class z_and_etheta_from_file():
    
    def __init__(self, file_path):
        archive = np.load(file_path, allow_pickle=True)
        for item in archive.files:
            self.array = archive[item]
    
    def sample(self):
        idx = random.randint(0, self.array.shape[0]-1)
        zlens, zsource, etheta = self.array[idx]
        return zlens, zsource, etheta
    
class UniformDist():
    
    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def sample(self):
        return random.uniform(self.lower_bound, self.upper_bound)
    
class EllipticityDist():
    
    def __init__(self, q_dist, theta_dist):
        self.q_dist = q_dist
        self.theta_dist = theta_dist

    def sample(self):
        return param_util.phi_q2_ellipticity(phi=self.theta_dist.sample(), q=self.q_dist.sample())

def generate_sample_config(
    x1_distribution,
    colour_distribution,
    magnitude_distribution,
    redshift_and_einsttheta_dist,
    ellipticity_dist,
    source_x_dist,
    source_y_dist
):
    # ------------------------------------------------------------------
    # Sampled Variables:
    # ------------------------------------------------------------------    
    z_lens, z_source, theta_e = redshift_and_einsttheta_dist.sample()
    e1, e2 = ellipticity_dist.sample()
    source_x = source_x_dist.sample() 
    source_y = source_y_dist.sample() 
    
    x1 =  x1_distribution.sample()
    colour = colour_distribution.sample()
    magnitude = magnitude_distribution.sample()
    
    phi, q = param_util.ellipticity2phi_q(e1, e2)
    
    sampled_vars = {
        "z_lens" : z_lens,
        "z_source" : z_source,
        "theta_e" : theta_e,
        "phi" : phi,
        "q" : q,
        "e1" : e1,
        "e2" : e2,
        "source_x" : source_x,
        "source_y" : source_y,
        "x1" : x1,
        "colour" : colour,
        "magnitude" : magnitude
    }
    
    # ------------------------------------------------------------------
    # Information needed to setup an instance of LensModel():
    # ------------------------------------------------------------------
    sie_params = {
        "theta_E" : theta_e,
        "e1" : e1,
        "e2" : e2,
        "center_x" : 0,
        "center_y" : 0
    }

    cosmology_params = {
        "H0" : 67.66,
        "Om0" : 0.309,
        "Ob0" : 0.05
    }

    lens_model_dict = {
        "model_kwargs" : {
            "lens_model_list" : ["SIE"],
            "z_lens" : z_lens,
            "z_source" : z_source,
            "cosmo" : FlatLambdaCDM(**cosmology_params)
        },
        "model_params" : [sie_params]
    }

    # ------------------------------------------------------------------
    # Information needed to setup two instances of LightModel():
    # ------------------------------------------------------------------

    host_galaxy_lightmodel_params = {
        "amp" : 0,#1000,
        "R_sersic" : 0.1, 
        "n_sersic" : 1.5,
        "center_x" : 0,
        "center_y" : 0
    }

    lens_galaxy_lightmodel_params = {
        "amp" : 0,#1000,
        "R_sersic" : 1,
        "n_sersic" : 3,
        "e1" : e1,
        "e2" : e2,
        "center_x" : 0,
        "center_y" : 0
    }

    light_model_dict = {
        "host_galaxy" : {
            "model_list" : ["SERSIC"],
            "model_params" : [host_galaxy_lightmodel_params]
        },
        "lens_galaxy" : {
            "model_list" : ["SERSIC_ELLIPSE"],
            "model_params" : [lens_galaxy_lightmodel_params]
        }
    }

    # ------------------------------------------------------------------
    # Information needed to setup an instance of PointSource():
    # ------------------------------------------------------------------
    source_params = {
            "ra_source" : source_x,
            "dec_source" : source_y,
            "point_amp" : 10000
            }
        
    point_source_dict = {
        "point_source_type_list" : ["LENSED_POSITION"],
        "source_params" : [source_params],
        "fixed_magnification_list" : [False] 
    }

    # ------------------------------------------------------------------
    # Information needed to setup an instance of SingleBase():
    # ------------------------------------------------------------------   
    imagedata_config = {
        "numPix" : 40,#int(theta_e*100),
        "deltaPix" : 0.11,
        "center_ra" : 0,
        "center_dec" : 0
    }

    singleband_dict = {
        "pixel_scale" : imagedata_config["deltaPix"],
        "exposure_time" : 55,
        "magnitude_zero_point" : 26.30,
        "read_noise" : 15.5,
        "ccd_gain": 2.3,
        "sky_brightness" : 22.93,
        "seeing" : 0.2,
        "num_exposures" : 1,
        "psf_type" : "Gaussian",
        "kernel_point_source" : None,
        "truncation" : 5,
        "data_count_unit" : "e-",
        "background_noise" : None     
    }
    
    noise_dict = {
        "exp_time" : 55,
        "background_rms" : SingleBand(**singleband_dict).background_noise
    }
    #print(noise_dict["background_rms"])

    # ------------------------------------------------------------------
    # Information needed to setup an instance of PixelGrid():
    # ------------------------------------------------------------------
    pixel_grid_dict = {
        "nx" : imagedata_config["numPix"],
        "ny" : imagedata_config["numPix"],
        "ra_at_xy_0" : -2.5,
        "dec_at_xy_0" : -2.5,
        "transform_pix2angle": np.array([[1,0],[0,1]]) * imagedata_config["deltaPix"]
    }

    # ------------------------------------------------------------------
    # Information needed to setup an instance of PSF():
    # ------------------------------------------------------------------
    psf_dict = {
        "psf_type" : "GAUSSIAN",
        "fwhm" : 0.073,
        "pixel_size" : imagedata_config["deltaPix"]
    }

    # ------------------------------------------------------------------
    # Information needed to setup an instance of ImageModel():
    # ------------------------------------------------------------------
    image_model_dict = {
        "kwargs_numerics" : {
            "supersampling_factor" : 1,
            "supersampling_convolution" : False
        }
    }

    event_model_config = {
        "lens_model_dict" : lens_model_dict,
        "light_model_dict" : light_model_dict,
        "source_dict" : point_source_dict,
        "pixel_grid_dict" : pixel_grid_dict,
        "psf_dict" : psf_dict,
        "image_model_dict" : image_model_dict
    }

    
    # ------------------------------------------------------------------
    # Information needed to configure Salt3:
    # ------------------------------------------------------------------
    model_info = {
    "source" : "salt3",
    "effects" : [sncosmo.CCM89Dust()],
    "effect_names" : ["host"],
    "effect_frames" : ["rest"]
    }

    params_info = {
        "z" : z_source,
        "t0" : 0,
        "x0" : source_params["point_amp"],
        "x1" :x1,
        "c" : colour,
        "m" : magnitude
    }

    lightcurve_model_config = {
        "model_dict" : model_info,
        "model_params_dict" : params_info
    }

    simulation_config = {
        "bands" : {"F087" : {"zp" : 26.30, "line_color": "turquoise"}},
        "time_range" : [-10,80],
        "time_precision" : 0.01,
        "source_plane" : "emission_plane",
        "time_delays" : None,
        "mag_cutoff" : 25.1
    }

    output_config = {
        "lightcurve_model_config" : lightcurve_model_config,
        "event_model_config" : event_model_config,
        "simulation_config" : simulation_config,
        "lens_model_dict" : lens_model_dict,
        "point_source_dict" : point_source_dict,
        "light_model_dict" : light_model_dict,
        "imagedata_config" : imagedata_config,
        "psf_dict" : psf_dict,
        "image_model_dict" : image_model_dict,
        "noise_config" : noise_dict
    }

    return output_config, sampled_vars
