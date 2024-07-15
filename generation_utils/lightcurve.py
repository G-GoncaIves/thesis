import sncosmo
import numpy as np
from scipy import stats

class LightCurve():

	# Source: https://smtn-002.lsst.io/#change-record
	implemented_bands_dict = {
	    "lsstu" : {"zp" : 27.03, "line_color": "blue"},
	    "lsstg" : {"zp" : 28.38, "line_color": "turquoise"},
	    "lsstr" : {"zp" : 28.16, "line_color": "green"},
	    "lssti" : {"zp" : 27.85, "line_color": "yellow"},
	    "lsstz" : {"zp" : 27.46, "line_color": "red"},
	    "lssty" : {"zp" : 26.68, "line_color": "mediumorchid"},
		"F062" : {"zp" : 26.56, "line_color": "blue"},
		"F087" : {"zp" : 26.30, "line_color": "turquoise"},
		"F106" : {"zp" : 26.44, "line_color": "green"},
		"F129" : {"zp" : 26.40, "line_color": "yellow"},
		"F158" : {"zp" : 26.43, "line_color": "red"},
		"F184" : {"zp" : 25.95, "line_color": "mediumorchid"},
		"F146" : {"zp" : 26.65, "line_color": "grey"}
	} 

	def __init__(
     self, 
     model_dict, 
     model_params_dict, 
     verbose=False, 
     implemented_bands_dict=implemented_bands_dict
     ):

		self.verbose = verbose
		self.implemented_bands_dict = implemented_bands_dict
		self.light_curve_model = sncosmo.Model(**model_dict)

		_params_dict = {}
		x0_is_none = model_params_dict["x0"] is None
		assert all([model_params_dict[f"{name}"] is not None for name in ["z", "t0", "x1", "c"]])
		
		for param_name, param in model_params_dict.items():
			
			if param_name == "m": # Because you don't want to set the magnitude as a model parameter
				pass

			elif param_name == "x0" and x0_is_none: # If x0 is not given, it is infereed from the peak magnitude
				pass

			else: # If all model parameters are given, just set them to the model
				try:
					_params_dict[param_name] = float(param)

				except TypeError:
					_params_dict[param_name] = param()

		self.light_curve_model.set(**_params_dict)
		self.light_curve_model.set_source_peakabsmag(model_params_dict["m"], "bessellb", "ab")

		if x0_is_none:
			try:
				M = float(model_params_dict["m"])		
		
			except TypeError:
				M = model_params_dict["m"]()
			
			self.light_curve_model.set_source_peakabsmag(M, "lsstu", "ab")
			x0 = self.light_curve_model.get("x0")
			self.light_curve_model.set(x0=x0)

	def get_band_flux(self, band_name, time_range, time_num):

		assert band_name in self.implemented_bands_dict.keys()
		band_info_dict = self.implemented_bands_dict[band_name]

		_time_range = np.linspace(*time_range, num=time_num)

		try:
			return self.light_curve_model.bandflux(band_name, _time_range, zp=band_info_dict["zp"], zpsys="ab"), _time_range

		except ValueError:
			if self.verbose:
				print(f"[!] Bandpass {band_name} out of spectral range [5000, .., 27500] and thus ignored.")
			return None, _time_range

	def get_band_magnitude(self, band_name, time_range, time_num=500):

		assert band_name in self.implemented_bands_dict.keys()
		band_info_dict = self.implemented_bands_dict[band_name]

		_time_range = np.linspace(*time_range, num=time_num)

		try:
			return self.light_curve_model.bandmag(
       					band = band_name, 
            			magsys="ab", 
               			time=_time_range
                  	), _time_range

		except ValueError:
			if self.verbose:
				print(f"[!] Bandpass {band_name} out of spectral range [5000, .., 27500] and thus ignored.")
			return None, _time_range

if __name__ == "__main__":

    model_info = {
        "source" : "salt3",
        "effects" : [sncosmo.CCM89Dust()],
        "effect_names" : ["host"],
        "effect_frames" : ["rest"]
    }
    
    params_info = {
        "z" : 1.0,
        "t0" : 55000,
        "x0" : 1e-3,
        "x1" : 0.0,
        "c" : 0
    }
    
    lc_model = LightCurve(model_dict=model_info, model_params_dict=params_info)
    y_flux = lc_model.get_band_flux(band_name="F184", time_range=[55000-15, 55000+15], time_num=100)
