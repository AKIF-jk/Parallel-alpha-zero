from pybind11.setup_helpers import build_ext
from setuptools import Extension, setup
import pybind11

setup(
    name="othello_cpp",
    ext_modules=[
        Extension(
            "othello_cpp",
            sources=["othello/cpp/pybind_bridge.cpp"],
            include_dirs=[pybind11.get_include(), "othello/cpp"],
            language="c++",
            extra_compile_args=["-O3", "-march=native", "-std=c++17", "-pthread"],
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
