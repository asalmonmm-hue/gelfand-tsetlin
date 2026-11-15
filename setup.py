from setuptools import Extension, setup


setup(
    package_dir={"": "src"},
    packages=["gelfand_tsetlin"],
    ext_modules=[
        Extension(
            "gelfand_tsetlin._gt",
            [
                "src/gelfand_tsetlin/_gt.c",
                "src/gelfand_tsetlin/gt_coefficient.c",
                "src/gelfand_tsetlin/gt_inner_product.c",
            ],
            depends=[
                "src/gelfand_tsetlin/gt_coefficient.h",
                "src/gelfand_tsetlin/gt_inner_product.h",
            ],
        ),
    ],
)
