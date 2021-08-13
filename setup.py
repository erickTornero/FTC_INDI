import setuptools

setuptools.setup(
    name="ftc",
    version="0.0.1",
    author="Erick Tornero",
    author_email="erickdeivy01@gmail.com",
    description="Indi Control for Faulted Rotor in Quadrotors",
    url="https://github.com/erickTornero/FTC_INDI",
    scripts=[],
    packages=setuptools.find_packages(include=['ftc', 'ftc.*', 'wrapper']),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[ 
        'numpy >= 1.18.1',
        'tqdm>=4.46.0',
        'gym >= 0.12.0',
        'pyyaml >= 5.4.1',
        'rospkg >= 1.3.0'
    ]
)
