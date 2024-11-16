import itertools
import math
from typing import Set, Tuple

import numpy as np
import sympy
from triton.language import dtype


class BallisticComputer:
    GRAVITY = 9.81
    def __init__(self, drag_coefficient: float = 0.47, projectile_cross_section_area: float = ((6 * 1/1000)**2) * np.pi, air_density: float = 103.55 * 1000, muzzle_velocity: float = 48.768):
        '''
        The air resistance values are likely unused
        :param drag_coefficient: (Unitless) [air resistance]
        :param projectile_cross_section_area: (m^2) [air resistance]
        :param air_density: In Pascals (kg/(m*s^2)) [air resistance]
        :param muzzle_velocity: (m/s)
        '''
        self.air_density = air_density
        self.projectile_cross_section_area = projectile_cross_section_area
        self.drag_coefficient = drag_coefficient
        self.muzzle_velocity = muzzle_velocity

    def __create_equation(self, rel_y: float, rel_z: float, flight_time: sympy.NumberSymbol):
        gravity = sympy.RealNumber(BallisticComputer.GRAVITY)
        muzzle_velocity = sympy.RealNumber(self.muzzle_velocity)
        rel_y = sympy.RealNumber(rel_y)
        rel_z = sympy.RealNumber(rel_z)

        equations = [
            muzzle_velocity * -sympy.sqrt(1 - ((rel_z / muzzle_velocity) / flight_time) ** 2) * flight_time - (
                        gravity / 2) * flight_time ** 2 - rel_y,
            muzzle_velocity * sympy.sqrt(1 - ((rel_z / muzzle_velocity) / flight_time) ** 2) * flight_time - (
                        gravity / 2) * flight_time ** 2 - rel_y
        ]
        return equations

    def calculate_elevation(self, target_rel_y: float, target_rel_z: float) -> np.ndarray:
        '''
        Calculate pitch angle (degrees)
        :param target_rel_y: Vertical Distance relative (m)
        :param target_rel_z: Horizontal Distance relative (m)
        :return: Targeting Solutions [(Aim Elevation (radians), Flight Time (seconds))]
        '''

        if target_rel_z < 0:
            return np.empty(0)

        flight_time_symbol = sympy.symbols('flight_time')

        eqs = self.__create_equation(target_rel_y, target_rel_z, flight_time_symbol)

        solutions = np.array(list(itertools.chain(list(sympy.solveset(eq, flight_time_symbol, sympy.Reals)) for eq in eqs)), dtype=np.float64)

        solutions = solutions[solutions > 0]

        intermediate_terms = ((target_rel_y + solutions ** 2 * BallisticComputer.GRAVITY / 2) / solutions / self.muzzle_velocity)

        valid_solutions = np.logical_and(-1 <= intermediate_terms, intermediate_terms <= 1)

        return np.hstack((np.arcsin(intermediate_terms[valid_solutions].reshape((-1, 1))), solutions[valid_solutions].reshape((-1, 1))))


if __name__ == '__main__':
    def evaluate_projectile_motion(angle: float, flight_time: float, muzzle_velocity: float) -> (float, float) :
        v0x = muzzle_velocity * np.cos(angle)
        v0y = muzzle_velocity * np.sin(angle)
        vy = v0y - BallisticComputer.GRAVITY * flight_time
        final_x = v0x * flight_time
        final_y = (v0y + vy) * flight_time / 2
        return final_x, final_y

    computer = BallisticComputer()

    solutions = computer.calculate_elevation(-1, 10)
    print(solutions)
    print([evaluate_projectile_motion(angle, flight_time, computer.muzzle_velocity) for (angle, flight_time) in solutions])