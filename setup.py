from setuptools import setup

setup(name='f110_gym',
      version='0.2.1',
      author='Hongrui Zheng',
      author_email='billyzheng.bz@gmail.com',
      url='https://f1tenth.org',
      package_dir={'': 'gym'},
      install_requires=['gym==0.19.0',
		            'numpy==1.22.0',
                        'Pillow==10.4.0',
                        'scipy>=1.10.1',
                        'numba>=0.58.1',
                        'pyyaml>=6.0.1',
                        'pyglet==1.4.11',
                        'pyopengl==3.1.7',
                        'pybullet==3.2.5']
      )
