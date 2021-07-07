class StateSpaceRobots:
    def __init__(self, names):
        self.names  =   names
        self._init_info_state_spaces()

    
    def _init_info_state_spaces(self):
        self.states_dict =   {
            'rotation_matrix': 9,
            'position': 3,
            'euler': 3,
            'quaternion': 4,
            'linear_vel': 3,
            'angular_vel': 3,
        }

    def get_state_space_shape(self):
        shape   =   0
        for name in self.names:
            try:
                shape += self.states_dict[name]
            except KeyError as error:
                raise Exception('Invalid name <{}> for state space'.format(name))

        return (shape, ) 

    
    def get_state_space_indexes(self):
        indexes =   {}
        index   =   0
        for name in self.names:
            try:
                indexes[name]   =   {
                    'start':    index,
                    'end':      index + self.states_dict[name],
                }
                index += self.states_dict[name] 
            except KeyError as error:
                raise Exception('Invalid name <{}> for state space'.format(name))

        return indexes