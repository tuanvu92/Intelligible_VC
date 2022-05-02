import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='intelligible_vc',
    version='0.0.1',
    author='Ho Tuan Vu',
    author_email='tuanvu@jaist.ac.jp',
    description='Intelligibility Enhancement Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://bitbucket.org/vinbdi-slp/vc_package/',
    project_urls={
        "Bug Tracker": "https://bitbucket.org/vinbdi-slp/vc_package/issues"
    },
    license='MIT',
    packages=['intelligible_vc'],
    install_requires=['torch',
                      'torchaudio',
                      'numpy==1.22.3',
                      'librosa==0.8.1',
                      'SoundFile',
                      'tensorboardX==2.1',
                      'tensorflow==2.4.1',
                      'tensorboard==2.4.1',
                      'munch==2.5.0'],
)
