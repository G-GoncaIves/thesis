# Import custom classes:
from lightcurve import LightCurve

# Import from standard libraties:
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Cosmo.background import Background
from lenstronomy.Util.data_util import magnitude2cps
import numpy as np

class SLEvent():

    def __init__(self, lightcurve_model_config, lensevent_model_config, simulation_config):

        expected_lensevent_model_config_params = [
            "lens_model_dict", 
            "light_model_dict", 
            "source_dict", 
            "pixel_grid_dict", 
            "psf_dict", 
            "image_model_dict"
        ]
        expected_lightcurve_model_config_params = [
            "model_dict", 
            "model_params_dict"
        ]
        expected_simulation_config_params = [
            "bands", 
            "time_range", 
            "time_precision", 
            "source_plane", 
            "time_delays"
        ]
        assert all([param in lensevent_model_config.keys() for param in expected_lensevent_model_config_params])
        assert all([param in lightcurve_model_config.keys() for param in expected_lightcurve_model_config_params])
        assert all([param in simulation_config.keys() for param in expected_simulation_config_params])
        self.banfluxes = self.simulate_source_bandfluxes(
            lightcurve_model_config=lightcurve_model_config, 
            bands_dict=simulation_config["bands"], 
            time_range=simulation_config["time_range"], 
            time_precision=simulation_config["time_precision"]
            )
        self.z_lens = lensevent_model_config["lens_model_dict"]["model_kwargs"]["z_lens"]
        self.z_source = lensevent_model_config["lens_model_dict"]["model_kwargs"]["z_source"]
        self.lens_model = LensModel(**lensevent_model_config["lens_model_dict"]["model_kwargs"])
        self.lens_model_parameters = lensevent_model_config["lens_model_dict"]["model_params"]
        self.image_details = self.convert_to_image_plane(lensevent_model_config)
        self.time_delayed_images_details = self.simulate_images(cutoff=simulation_config["mag_cutoff"])

    def simulate_source_bandfluxes(self, lightcurve_model_config, bands_dict, time_range, time_precision):
        """This method simulates a given source's lightcurves in every band.

        Args:
            lightcurve_model_config (dict): Dictionary with the parameters need to instanciate lenstronomy's LightCurve() class;
            bands_dict (dict): Dictionary with the information regarding the bands that will be simulated;
            time_range (int): Number of days the observation goes on for.
            time_precision (float): The maximum uncertainty between the time differences of two adjacent array elements.
   
        Returns:
            Dictionary with the simulated fluxes in all bands.
        """

        self.light_curve_model = LightCurve(**lightcurve_model_config)
        self.emission_start, self.emission_end = self._get_time_window(
            peak = lightcurve_model_config["model_params_dict"]["t0"], 
            time_range = time_range
        )
        self.time_num = int((self.emission_end - self.emission_start ) / time_precision) 
        self.time_precision = time_precision 
        self.emission_details = {
            "emission_start" : self.emission_start,
            "emission_end" : self.emission_end,
            "emission_peak" : lightcurve_model_config["model_params_dict"]["t0"]
        }
        self.source_bandfluxes = self._simulate_source_bandfluxes(bands_dict=bands_dict)
 
    def _simulate_source_bandfluxes(self, bands_dict):
        """This is a helper method to the class method 'simulate_bandfluxes()', that handles the actual simulation after 
          the preleminary setup has been done.

        Args:
            bands_dict (dict): Dictionary with the information regarding the bands that will be simulated.

        Returns:
            dict: Dictionary with the bandfluxes stored as arrays in the following way, dict['band_name'] = bandflux_array.
        """
        self.flux_max = 1
        band_fluxes = {}
        for band_name in bands_dict.keys():
            band_flux_array, band_flux_time = self._simulate_source_bandflux(band_name=band_name)
            band_fluxes[f"{band_name}"] = {
                "flux" : band_flux_array,
                "time" : band_flux_time,
                "band_zp" : bands_dict[band_name]["zp"]
            }
            band_flux_max = np.max(band_flux_array)
            if band_flux_max > self.flux_max:
                self.flux_max = band_flux_max
        return band_fluxes

    def _simulate_source_bandflux(self, band_name):
        """Helper method to '_simulate_bandfluxes' that handles the simulation of the flux for a given band.

        Args:
            band_name (str): Name of the band to be used for simulation.

        Returns:
            numpy.array: bandflux array.
        """ 
        flux_array, time_array = self.light_curve_model.get_band_flux(
            band_name = band_name, 
            time_range = [self.emission_start, self.emission_end], 
            time_num = self.time_num
        )
        if flux_array is None:
            return np.zeros(self.time_num), time_array
        else:
            return np.asarray(flux_array), time_array

    def _get_time_window(self, peak, time_range):
        """
        Args:
            peak (int): Day on which the maximum is located;
            time_range (int or tuple): Relative time observed before and after the peak. If int, both are consired the same.

        Returns:
            list: Days on which the observation started and ended.
        """
        try:
            time_observed_before, time_observed_after = int(time_range), int(time_range)
        except TypeError:
            time_observed_before, time_observed_after = time_range 
        return [peak+time_observed_before, peak+time_observed_after]

    def convert_to_image_plane(self, lensevent_model_config):
        """Converts positions in the host galaxy plane to the image plane.

        Args:
            lensevent_model_config (dict): Dictionary with the configuration required for lenstronoy's LensModel().

        Returns:
            dict: Dictionary with the positions, and corresponding time delays, of the images in the image plane.
        """
        source_dict = lensevent_model_config["source_dict"]["source_params"][0]
        source_plane_coordinates = [source_dict["ra_source"], source_dict["dec_source"]]
        self._find_image_positions(source_plane_coordinates)
        images_details_dict = {
            "ra"  : self.positions["theta_ra"],
            "dec" : self.positions["theta_dec"],
            "time_delay" : self._get_time_delays(),
            "magnifications" : self._get_magnifications(),
            "time_delay_distance" : self._get_ddt(),
            "kappa" : self._get_kappas()
        }
        return images_details_dict
        
    def _find_image_positions(self, position):
        """Helper method to 'convert_to_image_plane'. Actually handles the convertion after the prerequisites have been handled.

        Args:
            position (list): Position of the source in the host galaxy plane, structured like [ra_hostplane, dec_hostplane]
        """
        beta_ra, beta_dec = position
        theta_ra, theta_dec = self._source_to_image(
            beta_ra, 
            beta_dec
        )
        self.positions = {
            "theta_ra" : theta_ra,
            "theta_dec" : theta_dec,
            "beta_ra" : beta_ra,
            "beta_dec" : beta_dec
        }

    def _source_to_image(self, beta_ra, beta_dec):
        """Helper method to '_find_image_plane'. Solves the lens equation to obtain the positions in the image plane.

        Args:
            beta_ra (float): Source's position in the host plane;
            beta_dec (float): Source's position in the host plane.

        Returns:
            list: Images' positions in the image plane, strucutred like [[ra_multiple_images], [dec_multiple_images]]. Note that since there, possibly, multiple images each element of the output list is a list itself.
        """
        solver = LensEquationSolver(self.lens_model)
        return solver.findBrightImage(
            beta_ra, 
            beta_dec, 
            kwargs_lens = self.lens_model_parameters
        )

    def _get_time_delays(self):
        """Helper method to 'convert_to_image_plane'. Calculates the time delays for all the images.

        Returns:
            nupy.array: Array with the time delays of all simulated images.
        """

        nbr_images = len(self.positions["theta_ra"])
        time_delay_array = np.zeros(nbr_images)
        for n in range(nbr_images):
            ra = self.positions["theta_ra"][n] 
            dec = self.positions["theta_dec"][n]
            time_delay = self.lens_model.arrival_time(
                x_image = ra, 
                y_image = dec, 
                kwargs_lens = self.lens_model_parameters
            )
            _, decimals_str = str(self.time_precision).split(".")
            time_delay_array[n] = np.round(
                time_delay, 
                decimals = len(decimals_str)
            )
        time_delay_array = time_delay_array - np.min(time_delay_array)
        return time_delay_array 

    def _get_ddt(self):
        bg = Background()
        td_distances = []
        for ra, dec in zip(self.positions["theta_ra"], self.positions["theta_dec"]):
            td_distance = bg.ddt(
                z_lens = self.z_lens,
                z_source= self.z_source
            )
            td_distances.append(td_distance)
        return td_distances
           
    def _get_magnifications(self):
        
        magnifications = []
        for ra, dec in zip(self.positions["theta_ra"], self.positions["theta_dec"]):
            image_magnification = self.lens_model.magnification(x=ra, y=dec, kwargs=self.lens_model_parameters)
            magnifications.append(image_magnification)
        
        return magnifications

    def _get_kappas(self):
        
        kappas = []
        for ra, dec in zip(self.positions["theta_ra"], self.positions["theta_dec"]):
            image_kappas = self.lens_model.kappa(x=ra, y=dec, kwargs=self.lens_model_parameters)
            kappas.append(image_kappas)
        
        return kappas

    def simulate_images(self, cutoff):
        """Simulates the all the image's bandflux, by shifting the simulated source bandflux in accordance to each of the image's time delay.

        Returns:
            dict: Dictionary with all the images' simulated bandfluxes. Structured like dict['image_ra, image_dec'] = image_bandflux. Note the positions here are in the image plane.
        """
        images_info_dict = {}
        nbr_images = len(self.positions["theta_ra"])
        for n in range(nbr_images):
            self._update_image_badfluxes_details(
                details_dict = images_info_dict,
                image_idx = n,
                cutoff = cutoff
            )
        self.images_info_dict = images_info_dict
        
    def _update_image_badfluxes_details(self, details_dict, image_idx, cutoff):
        """Helper method to 'simulate_images_bandfluxes'.

        Args:
            details_dict (dict): Dictionary where to store the image's bandflux;
            image_idx (int): idx to identify the image.
        """

        image_ra = self.image_details["ra"][image_idx]
        image_dec = self.image_details["dec"][image_idx]
        image_time_delay = self.image_details["time_delay"][image_idx]
        image_magnification = self.image_details["magnifications"][image_idx]
        image_bandflux = self._simulate_image_bandflux(
            magnification = image_magnification, 
            time_delay = image_time_delay,
            cutoff = cutoff
        )
        details_dict[f"{image_ra},{image_dec}"] = image_bandflux
        
    def _simulate_image_bandflux(self, magnification, time_delay, cutoff):
    
        image_bandfluxes_dict = {}          
        for band_name, bandflux_dict in self.source_bandfluxes.items():
            delayed_bandflux_dict = self._time_delay_bandflux(
                bandflux_dict = bandflux_dict, 
                delay = time_delay
            )
            magnified_bandlufx_dict = self._magnify_bandflux(
                bandflux_dict = delayed_bandflux_dict, 
                magnification = magnification
            )
            image_bandfluxes_dict[f"{band_name}"] = self._apply_cutoff(
                bandflux_dict = magnified_bandlufx_dict,
                cutoff = cutoff
            )
        return image_bandfluxes_dict
            
    def _apply_cutoff(self, bandflux_dict, cutoff):
        
        if cutoff is not None:
            flux_cutoff = magnitude2cps(
                magnitude = cutoff,
                magnitude_zero_point = bandflux_dict["band_zp"]
                )
            idxs_above_cutoff = np.where(
                bandflux_dict["flux"] > flux_cutoff
                )
            cutoff_dict = {
                "time" : bandflux_dict["time"][idxs_above_cutoff],
                "flux" : bandflux_dict["flux"][idxs_above_cutoff],
                "band_zp" : bandflux_dict["band_zp"]
            }
            return cutoff_dict
        else:
            return bandflux_dict
        
    def _time_delay_bandflux(self, bandflux_dict, delay):
        """Helper method to '_time_delay_bandfluxes'. 

        Args:
            bandflux_dict (dict) : Dictionary with the flux_array and the time_array of the corresponding band;
            delay_length (int):  Ammount by which to delay the corresponding bandflux.

        Returns:
            Numpy.array: Time delayed bandflux array.
        """

        delayed_bandflux_dict = {
            "flux" : bandflux_dict["flux"],
            "time" : bandflux_dict["time"] + delay,
            "band_zp" : bandflux_dict["band_zp"]
        }     
        return delayed_bandflux_dict
 
    def _magnify_bandflux(self, bandflux_dict, magnification):
        
        magnified_bandflux_dict = {
            "flux" : np.abs(magnification*bandflux_dict["flux"]),
            "time" : bandflux_dict["time"],
            "band_zp" : bandflux_dict["band_zp"]
        }
        return magnified_bandflux_dict
