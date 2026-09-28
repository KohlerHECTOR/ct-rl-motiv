"""Shared helpers to build HalfCheetah envs with a modified MuJoCo frametime."""
import os
import xml.etree.ElementTree as ET
from fractions import Fraction

import gymnasium as gym
import gymnasium.envs.mujoco as gym_mujoco
import mujoco

DEFAULT_XML = os.path.join(os.path.dirname(gym_mujoco.__file__), 'assets', 'half_cheetah.xml')
ROOT = os.path.dirname(os.path.abspath(__file__))
XML_DIR = os.path.join(ROOT, 'xmls')
MODEL_DIR = os.path.join(ROOT, 'models')

DEFAULT_FRAMETIME = 0.01
DEFAULT_MAX_STEPS = 1000
MULTS = [Fraction(1, k) for k in range(10, 1, -1)] + [Fraction(k) for k in range(1, 11)]  # 1/10, ..., 1/2, 1, 2, ..., 10


def mult_label(mult):
    """Filesystem-friendly label, e.g. '1-3' for 1/3 and '4' for 4."""
    return f'{mult.numerator}-{mult.denominator}' if mult.denominator != 1 else str(mult.numerator)


def frametime(mult):
    return DEFAULT_FRAMETIME * float(mult)


def max_steps(mult):
    """Episode length (in steps) that keeps the simulated episode duration constant."""
    return round(DEFAULT_MAX_STEPS / float(mult))


def model_path(mult, seed=0):
    return os.path.join(MODEL_DIR, f'ppo_mult{mult_label(mult)}_seed{seed}.zip')


def write_xml(mult):
    """Copy the default HalfCheetah model with a different MuJoCo timestep."""
    os.makedirs(XML_DIR, exist_ok=True)
    tree = ET.parse(DEFAULT_XML)
    tree.getroot().find('option').set('timestep', repr(frametime(mult)))
    path = os.path.join(XML_DIR, f'half_cheetah_mult{mult_label(mult)}.xml')
    # write-then-rename so parallel env processes never read a half-written file
    tmp = f'{path}.{os.getpid()}.tmp'
    tree.write(tmp)
    os.replace(tmp, path)
    return path


class InstabilityFlag(gym.Wrapper):
    """Sets info['unstable'] = True once MuJoCo has reported a NaN/Inf/huge value during the episode."""

    WARNINGS = [mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL]

    def _count(self):
        return sum(self.env.unwrapped.data.warning[w].number for w in self.WARNINGS)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._baseline = self._count()
        self._unstable = False
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._unstable |= self._count() > self._baseline
        info['unstable'] = self._unstable
        return obs, reward, terminated, truncated, info


def make_env(mult):
    env = gym.make('HalfCheetah-v5', xml_file=write_xml(mult), max_episode_steps=max_steps(mult))
    return InstabilityFlag(env)
