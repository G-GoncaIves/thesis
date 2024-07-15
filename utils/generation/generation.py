
from simulate import SLEvent
from sampler import generate_sample_config, NormalDist, AsymmetricGaussian, MagnitudeDist, UniformDist, EllipticityDist, z_and_etheta_from_file
from video import GenerateVideo
import pandas as pd

# Other imports:
import sncosmo
from astropy.cosmology import FlatLambdaCDM
import lenstronomy.Util.param_util as param_util
from lenstronomy.Util.data_util import cps2magnitude
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import date
from lenstronomy.SimulationAPI.observation_api import SingleBand

from scipy.stats import norm
import random

from scipy.stats import skewnorm


band = "F087"

def simulate_sample_system(sample_config):
    
    try:
        simulated_event = SLEvent(
            lightcurve_model_config=sample_config["lightcurve_model_config"],
            lensevent_model_config=sample_config["event_model_config"],
            simulation_config=sample_config["simulation_config"]
        )
        return simulated_event
    except ValueError:
        return None
    
c_mu_uncert          = NormalDist(mu=-0.043, sigma=0.01)
c_sigma_minus_uncert = NormalDist(mu=0.052 , sigma=0.006)
c_sigma_plus_uncert  = NormalDist(mu=0.107 , sigma=0.01)

x1_mu_uncert          = NormalDist(mu=0.945, sigma=0.1)
x1_sigma_minus_uncert = NormalDist(mu=1.553, sigma=0.117)
x1_sigma_plus_uncert  = NormalDist(mu=0.257, sigma=0.078)

reference_mag_uncert = NormalDist(mu=-19.4, sigma=0.12)

x1_distribution_config = {
    "mu" : x1_mu_uncert.sample(),
    "sigma_minus" : x1_sigma_minus_uncert.sample(),
    "sigma_plus" : x1_sigma_plus_uncert.sample(),
    "discretization" : 10000,
    "x_range" : [-1,2]
}
x1_dist = AsymmetricGaussian(**x1_distribution_config)

c_distribution_config = {
    "mu" : c_mu_uncert.sample(),
    "sigma_minus" : c_sigma_minus_uncert.sample(),
    "sigma_plus" : c_sigma_plus_uncert.sample(),
    "discretization" : 10000,
    "x_range" : [-0.3,0.3]   
}
c_dist = AsymmetricGaussian(**c_distribution_config)

magnitude_dist_config = {
    "alpha" : 3.1,
    "beta" : 3.1,
    "x1" : x1_dist.sample,
    "c" : c_dist.sample,
    "m_ref" : reference_mag_uncert.sample
}
magnitude_dist = MagnitudeDist(**magnitude_dist_config)

q = NormalDist(mu=0.7, sigma=0.15)
theta = UniformDist(lower_bound=-np.pi/2, upper_bound=np.pi/2)

z_and_etheta = z_and_etheta_from_file(file_path="sample_zl_zsn_theta.npz")
ellipticity = EllipticityDist(q_dist=q, theta_dist=theta)
source_x = UniformDist(lower_bound=-1, upper_bound=1)
source_y = UniformDist(lower_bound=-1, upper_bound=1)

def store_sample(dataframe, sample_dict, id, multiplicity, image_details, video_interval, observed_images_ratio, peak_flux, peak_mag):
    sample_dict["id"] = id
    sample_dict["multiplicity"] = multiplicity
    sample_dict["im_peak_mags"] = peak_mag
    sample_dict["im_peak_fluxs"] = peak_flux
    if image_details is not None:
        sample_dict["im_ra"] = image_details["ra"]
        sample_dict["im_dec"] = image_details["dec"]
        sample_dict["im_td"] = image_details["time_delay"]
        sample_dict["im_mag"] = image_details["magnifications"]
        sample_dict["im_tdd"] = image_details["time_delay_distance"]
        sample_dict["im_kappa"] = image_details["kappa"]
        sample_dict["video_interval"] = video_interval
        sample_dict["imgs_seen"] = observed_images_ratio
    else:
        sample_dict["im_ra"] = []
        sample_dict["im_dec"] = []
        sample_dict["im_td"] = []
        sample_dict["im_mag"] = []
        sample_dict["im_tdd"] = []
        sample_dict["im_kappa"] = []
        sample_dict["video_interval"] = []
        sample_dict["imgs_seen"] = 0
        
    dataframe.loc[len(dataframe)] = sample_dict
    
def check_if_visible(details_images, multiplicity, visibility_ratio, min_visibility_len):
    
    visible_count = 0
    for _, details in details_images.items():
        visible_in_all_bands = all([len(details[band]["flux"]) >= min_visibility_len for band in details.keys()])
        if visible_in_all_bands:
            visible_count += 1
    ratio = visible_count / multiplicity
    if ratio >= visibility_ratio:
        return True
    else:
        return False
    
def image_was_observed(start, stop, cadence, interval):
    observed_time_steps = np.arange(start, stop, step=cadence)
    return any([interval[0] <= i and interval[1] >= i for i in observed_time_steps ])
    
def analyse_image_timing(images_info_dict, cadence, band):
    
    timings = []
    observed_images, max_time, min_time = 0, 0, 0
    for n, (_, image_dict) in enumerate(images_info_dict.items()):
        start, stop = image_dict[band]["time"][0], image_dict[band]["time"][-1]
        timings.append((start, stop))
        if start < min_time:
            min_time =start 
        if stop > max_time:
            max_time = stop
    for image_timing in timings:
        if image_was_observed(
            start = min_time,
            stop = max_time,
            cadence = cadence,
            interval = image_timing
        ):
            observed_images += 1
    ratio = observed_images / (n+1)
    return min_time, max_time, ratio

def vars_in_desired_range(sampled_vars, image_details, numPix=40, deltaPix=0.05):
    c1 = sampled_vars["z_source"] > sampled_vars["z_lens"]
    c2 = all([td < 200 for td in image_details["time_delay"]])
    c3 = all([x < numPix*deltaPix and y < numPix*deltaPix for x,y in zip(image_details["ra"], image_details["dec"])])
    return c1 and c2 and c3

def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

def simulate_dataset(
    nbr, 
    store_dir, 
    name=None, 
    overwrite=False, 
    desired_multiplicity=2, 
    threshold=1000, 
    cadence=5, 
    save_every=500, 
    visibility_ratio=0,
    min_visibility_len=0,
    desc=""
    ):
    
    if name is None:
        current_date = date.today()
        name = current_date.strftime("%d-%m-%y")
    dataset_dir = os.path.join(store_dir, name, f"m_{desired_multiplicity}").replace("\\","/")
    try:
        os.mkdir(dataset_dir)
        timeseries_dir = os.path.join(dataset_dir, "Videos").replace("\\","/")
        os.mkdir(timeseries_dir)
    except FileExistsError:
        if not overwrite:
            print(f"Trying to store new dataset to already existing dir, please provide new path for storing or set 'overwrite' to True. \n\t\
                Current Provided Dir: {dataset_dir}")
            return False
        else:
            delete_files_in_directory(dataset_dir)
            timeseries_dir = os.path.join(dataset_dir, "Videos").replace("\\","/")
            os.mkdir(timeseries_dir)
    except FileNotFoundError:
        root_dir = os.path.join(store_dir, name).replace("\\","/")
        os.mkdir(root_dir)
        os.mkdir(dataset_dir)
        timeseries_dir = os.path.join(dataset_dir, "Videos").replace("\\","/")
        os.mkdir(timeseries_dir)
    
    dataframe_dict = {
        "id" : [],
        "multiplicity" : [],
        "z_lens" : [],
        "z_source" : [],
        "theta_e" : [],
        "phi" : [],
        "q" : [],
        "e1" : [],
        "e2" : [],
        "source_x" : [],
        "source_y" : [],
        "x1" : [],
        "colour" : [],
        "magnitude" : [],
        "im_ra" : [],
        "im_dec" : [],
        "im_td" : [],
        "im_mag" : [],
        "im_tdd" : [],
        "im_kappa" : [],
        "video_interval" : [],
        "imgs_seen" : [],
        "im_peak_mags" : [],
        "im_peak_fluxs" : []
    }
    simulation_df = pd.DataFrame(
        dataframe_dict, 
        dtype=object
    )
    type1_errors_df = pd.DataFrame(
        dataframe_dict, 
        dtype=object
    )
    type2_errors_df = pd.DataFrame(
        dataframe_dict, 
        dtype=object
    )
    n = k = 0
    tries_pbar = tqdm(
        total = threshold, 
        position = 0, 
        desc = "Number of Tries", 
        colour = "blue",
	leave = True
    )
    finds_pbar = tqdm(
        total = nbr, 
        position = 1, 
        desc = "Systems Found ", 
        colour = "magenta",
	leave = True
    )
    while n < nbr and k < threshold:
        sample_config, sampled_vars = generate_sample_config(
            x1_distribution=x1_dist,
            colour_distribution=c_dist,
            magnitude_distribution=magnitude_dist,
            redshift_and_einsttheta_dist=z_and_etheta,
            ellipticity_dist=ellipticity,
            source_x_dist=source_x,
            source_y_dist=source_y
        )
        if sampled_vars["z_source"] > sampled_vars["z_lens"]:
            system = simulate_sample_system(sample_config=sample_config)
            if system is not None:
                multiplicity = len(system.image_details["ra"])
                if multiplicity == desired_multiplicity:
                    if check_if_visible(
                        details_images = system.images_info_dict,
                        multiplicity = multiplicity,
                        visibility_ratio = visibility_ratio,
                        min_visibility_len = min_visibility_len 
                    ):
                        video_start, video_stop, images_seen_ratio = analyse_image_timing(
                            system.images_info_dict, 
                            cadence=cadence, 
                            band="F087"
                        )
                        try:
                            gen_video = GenerateVideo(
                                config_lensmodel = sample_config["lens_model_dict"], 
                                config_pointsource = sample_config["point_source_dict"], 
                                config_lightmodel = sample_config["light_model_dict"],
                                config_noise = sample_config["noise_config"],
                                config_imagedata = sample_config["imagedata_config"],
                                config_psf = sample_config["psf_dict"],
                                config_imagemodel = sample_config["image_model_dict"],
                                details_images = system.images_info_dict
                            )
                            system_time_delays = system.image_details["time_delay"]
                            max_time_delay = 190#np.max(system_time_delays)
                            video = gen_video.simulate_video(
                                band="F087", 
                                observation_window=[-10,max_time_delay], 
                                observation_cadence=cadence
                            )
                            no_td_details = gen_video.get_no_td_image_details(band="F087")
                            no_td_frame = gen_video.simulate_no_td_frame(
                                band="F087",
                                peak_details = no_td_details
                            )
                            if vars_in_desired_range(
                                sampled_vars = sampled_vars,
                                image_details = system.image_details
                            ):
                                sample_id = (len(str(nbr))-len(str(n)))*"0" + f"{n}"
                                video_path = os.path.join(
                                    dataset_dir, 
                                    "Videos", 
                                    f"{sample_id}"
                                ).replace("\\","/")
                                no_td_frame_path = os.path.join(
                                    dataset_dir, 
                                    "Videos", 
                                    f"{sample_id}_no_td"
                                ).replace("\\","/")
                                np.save(
                                    file=video_path, 
                                    arr=video
                                )
                                np.save(
                                    file=no_td_frame_path, 
                                    arr=no_td_frame
                                )
                                im0_peak_flux, im1_peak_flux = no_td_details["point_amp"]
                                im0_peak_mag = cps2magnitude(im0_peak_flux, magnitude_zero_point=26.30)
                                im1_peak_mag = cps2magnitude(im1_peak_flux, magnitude_zero_point=26.30)
                                store_sample(
                                    dataframe=simulation_df, 
                                    sample_dict=sampled_vars.copy(), 
                                    id=sample_id, 
                                    multiplicity=multiplicity,
                                    image_details = system.image_details, 
                                    video_interval = [video_start, video_stop], 
                                    observed_images_ratio = images_seen_ratio,
                                    peak_flux = [im0_peak_flux, im1_peak_flux],
                                    peak_mag = [im0_peak_mag, im1_peak_mag]
                                )
                                n += 1
                                finds_pbar.update(1)
                        except IndexError:
                            store_sample(
                                dataframe=type1_errors_df, 
                                sample_dict=sampled_vars.copy(), 
                                id="", 
                                multiplicity=multiplicity,
                                image_details = system.image_details, 
                                video_interval = [video_start, video_stop], 
                                observed_images_ratio = images_seen_ratio
                            )
                        if n % save_every == 0:
                            simulated_pickle_path = os.path.join(dataset_dir, f"{name}_simulated.pickle").replace("\\","/")
                            type1_errors_pickle_path = os.path.join(dataset_dir, f"{name}_type1_errors.pickle").replace("\\","/")
                            type2_errors_pickle_path = os.path.join(dataset_dir, f"{name}_type2_errors.pickle").replace("\\","/")
                            simulation_df.to_pickle(
                                simulated_pickle_path
                            )
                            type1_errors_df.to_pickle(
                                type1_errors_pickle_path
                            )
                            type2_errors_df.to_pickle(
                                type2_errors_pickle_path
                            )
        else:
            multiplicity = "Nan"
            sample_id = (len(str(nbr))-len(str(n)))*"0" + f"{n}"
            store_sample(
                dataframe=type2_errors_df, 
                sample_dict=sampled_vars.copy(), 
                id=sample_id, 
                multiplicity=multiplicity,
                image_details = None,
                video_interval= None,
                observed_images_ratio=None,
                peak_flux = None,
                peak_mag = None
            )
        tries_pbar.update(1)
        k += 1
    tries_pbar.close()
    finds_pbar.close()
    return True
        
