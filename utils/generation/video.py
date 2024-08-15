# Imports from the Lenstronomy library:
# https://lenstronomy.readthedocs.io/en/latest/
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.PointSource.point_source import PointSource
from lenstronomy.LightModel.light_model import LightModel
import lenstronomy.Util.param_util as param_util
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Util.simulation_util import data_configure_simple
import lenstronomy.Util.image_util as image_util


# Imports from other Libraries:
import numpy as np
from astropy.cosmology import FlatLambdaCDM


class SetupGeneration():
    
    def __init__(
        self,
        config_lensmodel,
        config_lightmodel,
        config_pointsource,
        config_psf,
        config_imagedata,
        config_imagemodel,
        details_images,
        config_noise
        ):
        self.lensmodel = LensModel(**config_lensmodel["model_kwargs"])
        self.host_galaxy_lighmodel = LightModel(config_lightmodel["host_galaxy"]["model_list"])
        self.lens_galaxy_lighmodel = LightModel(config_lightmodel["lens_galaxy"]["model_list"])
        self.pointsource = PointSource(
            point_source_type_list = config_pointsource["point_source_type_list"],
            fixed_magnification_list = config_pointsource["fixed_magnification_list"],
            lens_model = self.lensmodel
        )
        self.psf = PSF(**config_psf)
        config_imagedata.update({"background_rms": config_noise["background_rms"]})
        _config_imagedata = data_configure_simple(**config_imagedata)
        self.imagedata = ImageData(**_config_imagedata)
        self.numpix = config_imagedata["numPix"]
        self.imagemodel = ImageModel(**{
            "data_class" : self.imagedata,
            "psf_class" : self.psf,
            "lens_model_class" : self.lensmodel,
            "source_model_class" : self.host_galaxy_lighmodel,
            "lens_light_model_class" : self.lens_galaxy_lighmodel,
            "point_source_class" : self.pointsource,
            "kwargs_numerics" : config_imagemodel["kwargs_numerics"]
        })
        self.image_parameters = {
            "kwargs_lens" : config_lensmodel["model_params"], 
            "kwargs_source" : config_lightmodel["host_galaxy"]["model_params"], 
            "kwargs_lens_light" : config_lightmodel["lens_galaxy"]["model_params"]
        }
        self.details_images = details_images
        self.config_noise = config_noise
        
    @property
    def details_images(self):
        return self._details_images
        
    @details_images.setter
    def details_images(self, details : dict):
        for position, details_image in details.items():
            ra_str, dec_str = position.split(",")
            try:
                ra, dec = float(ra_str), float(dec_str)
            except ValueError:
                print(f"Problems with the sources' configuration. \n\t Invalid Position String: \n\t\t Got: '{position}' \n\t\t Expected: 'ra,dec'")
            
            for key in details_image.keys():
                try:
                    np.asarray(details_image[key]["flux"], dtype=float)
                    np.asarray(details_image[key]["time"], dtype=float)
                except ValueError:
                    print(f"Problems with the sources' configuration. \n\t Unexpected array provided for {key} associated with source at {position}. Please provide an array of floats (or variables castable to floats).")
        self._details_images = details
        
class GenerateFrame(SetupGeneration):
    
    def __init__(
        self,
        config_lensmodel,
        config_lightmodel,
        config_pointsource,
        config_psf,
        config_imagedata,
        config_imagemodel,
        details_images,
        config_noise
        ):
        super().__init__(
            config_lensmodel,
            config_lightmodel,
            config_pointsource,
            config_psf,
            config_imagedata,
            config_imagemodel,
            details_images,
            config_noise
        )
        
    def simulate_current_frame(self, time, band, custom_details=None):
        if custom_details is not None:
            current_source_details = custom_details
        else:
            current_source_details = self.get_current_image_details(
                time=time, 
                band=band
            )
        current_image_parameters = self.image_parameters
        current_image_parameters["kwargs_ps"] = [current_source_details]
        image = self.imagemodel.image(**current_image_parameters)
        poisson = image_util.add_poisson(image, exp_time=self.config_noise["exp_time"])
        background = image_util.add_background(image, sigma_bkd=self.config_noise["background_rms"])
        return image + poisson + background
    
    def simulate_no_td_frame(self, band, peak_details=None):
        if peak_details is None:
            peak_details = self.get_no_td_image_details(
                band=band
            )
        current_image_parameters = self.image_parameters
        current_image_parameters["kwargs_ps"] = [peak_details]
        image = self.imagemodel.image(**current_image_parameters)
        poisson = image_util.add_poisson(image, exp_time=self.config_noise["exp_time"])
        background = image_util.add_background(image, sigma_bkd=self.config_noise["background_rms"])
        return image + poisson + background

    def calculate_snr(self, peak_details):
        current_image_parameters = self.image_parameters
        current_image_parameters["kwargs_ps"] = [peak_details]
        image = self.imagemodel.image(**current_image_parameters)
        poisson = image_util.add_poisson(image, exp_time=self.config_noise["exp_time"])
        background = image_util.add_background(image, sigma_bkd=self.config_noise["background_rms"])
        noisy_image = image + poisson + background
        avg_noise = np.mean(np.abs(noisy_image - image)) 
        return peak_details["point_amp"][0] / avg_noise
        
    def get_current_image_details(self, time, band):
        image_details = {
            "ra_image" : [],
            "dec_image" : [],
            "point_amp" : []
        }
        for position, source_config in self.details_images.items():
            ra, dec = position.split(",")
            ra, dec = float(ra), float(dec)
            time_array = source_config[band]["time"]
            flux_idx = self._flux_idx_at_given_time(
                current_time=time, 
                time_array=time_array
            )
            if flux_idx is None:
                current_band_flux = 0
            else:
                flux_array = source_config[band]["flux"]
                current_band_flux = flux_array[flux_idx]
            image_details["ra_image"].append(ra)
            image_details["dec_image"].append(dec)
            image_details["point_amp"].append(current_band_flux)
        return image_details
    
    def get_no_td_image_details(self, band):
        image_details = {
            "ra_image" : [],
            "dec_image" : [],
            "point_amp" : []
        }
        for position, source_config in self.details_images.items():
            ra, dec = position.split(",")
            ra, dec = float(ra), float(dec)
            flux_array = source_config[band]["flux"]
            peak_band_flux = np.max(flux_array)
            image_details["ra_image"].append(ra)
            image_details["dec_image"].append(dec)
            image_details["point_amp"].append(peak_band_flux)
        return image_details
        
    def _flux_idx_at_given_time(self, current_time, time_array):
        if current_time >= time_array[0] and current_time <= time_array[-1]:
            return (np.abs(time_array - current_time)).argmin()
        else:
            return None

class GenerateVideo(GenerateFrame):

    def __init__(
        self,
        config_lensmodel,
        config_lightmodel,
        config_pointsource,
        config_psf,
        config_imagedata,
        config_imagemodel,
        details_images,
        config_noise
        ):
        super().__init__(
            config_lensmodel,
            config_lightmodel,
            config_pointsource,
            config_psf,
            config_imagedata,
            config_imagemodel,
            details_images,
            config_noise,
        )

    def simulate_video(self, band, observation_window=[0,60], observation_cadence=5):
        array_xdim = array_ydim = self.numpix
        time_array = np.arange(
            start = observation_window[0], 
            stop = observation_window[1], 
            step = observation_cadence
        )
        video = np.zeros(
            (len(time_array), array_ydim, array_xdim)
        )
        for n, time in enumerate(time_array):
            current_frame = self.simulate_current_frame(
                time = time, 
                band = band
            )
            video[n] = current_frame
        return video
        
