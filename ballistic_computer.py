import math
from typing import Set, Tuple

import numpy as np
import sympy


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

        equation = rel_y - (muzzle_velocity * sympy.sqrt(1 - ((rel_z / muzzle_velocity) / flight_time) ** 2) * flight_time) + (flight_time ** 2) * gravity / 2

        return equation

    def calculate_elevation(self, target_rel_y: float, target_rel_z: float) -> Set[Tuple[float, float]]:
        '''
        Calculate pitch angle (degrees)
        :param target_rel_y: Vertical Distance relative (m)
        :param target_rel_z: Horizontal Distance relative (m)
        :return: Targeting Solutions [(Aim Elevation (radians), Flight Time (seconds))]
        '''

        flight_time_symbol = sympy.symbols('flight_time')

        eq = self.__create_equation(target_rel_y, target_rel_z, flight_time_symbol)

        solutions = {(math.acos(target_rel_z / flight_time / self.muzzle_velocity), flight_time) for flight_time in sympy.solveset(eq, flight_time_symbol, sympy.Reals) if -1 < (target_rel_z / flight_time / self.muzzle_velocity) < 1}

        return solutions


if __name__ == '__main__':
    def evaluate_projectile_motion(angle: float, flight_time: float, muzzle_velocity: float) -> (float, float) :
        v0x = muzzle_velocity * np.cos(angle)
        v0y = muzzle_velocity * np.sin(angle)
        vy = v0y - BallisticComputer.GRAVITY * flight_time
        final_x = v0x * flight_time
        final_y = (v0y + vy) * flight_time / 2
        return final_x, final_y

    computer = BallisticComputer()

    solutions = computer.calculate_elevation(1, 10)
    print(solutions)
    print([evaluate_projectile_motion(angle, flight_time, computer.muzzle_velocity) for (angle, flight_time) in solutions])