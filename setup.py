from setuptools import Extension, setup


setup(
    package_dir={"": "src"},
    packages=["gelfand_tsetlin"],
    ext_modules=[
        Extension(
            "gelfand_tsetlin._gt",
            ["src/gelfand_tsetlin/_gt.c", "src/gelfand_tsetlin/gt_coefficient.c"],
            depends=["src/gelfand_tsetlin/gt_coefficient.h"],
        ),
    ],
)
