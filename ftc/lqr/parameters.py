import json
class Parameters:
    def __init__(self, quad_params_path=None, control_params_path=None):
        if quad_params_path is None or control_params_path is None:
            raise NotImplementedError("")
        else:
            parameters = self.load_params(quad_params_path, control_params_path)
            self.freq = parameters['freq'] # control frequency

            self.fail_id =  parameters['fail_id']

            self.b = parameters['body_width']    # b -> key [m]
            self.l = parameters['body_height']    # l -> key
            self.Ix = parameters['Ix']    # [kg m^2]
            self.Iy = parameters['Iy']
            self.Iz = parameters['Iz']
            self.mass = parameters['mass']   # [kg]
            self.g = parameters['gravity']   #key -> g

            self.k0 = parameters['motor_constant']    # k0 key -> propeller thrust coefficient
            self.t0 = parameters['moment_constant']    # t0 key -> torque coefficient
            self.w_max = parameters['w_max']   # max / min propeller rotation rates, [rad/s]
            self.w_min = parameters['w_min']

            raise NotImplementedError("Load here your control parameters gven in control_params_path, if you dont need, please remove this line")
    
    @property
    def gravity(self):
        return self.g

    def load_params(self, quad_params_fp, control_params_fp):
        q_params = self._read_quad_params(quad_params_fp)
        c_params = self._read_control_params(control_params_fp)
        return {**q_params, **c_params}

    def _read_quad_params(self, quad_params_fp):
        return self._open_file(quad_params_fp)

    def _read_control_params(self, control_params_fp):
        return self._open_file(control_params_fp)

    def _open_file(self, file_path):
        with open(file_path, 'r') as fp:
            return json.load(fp)