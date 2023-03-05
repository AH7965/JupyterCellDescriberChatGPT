import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="JupyterCellDescriberChatGPT",
    version="0.1.0",
    author="Akira Hashimoto",
    author_email="hashimoto7965@gmail.com",
    description="Using the OpenAI GPT-3 model, describe the process being performed in a cell in Jupyter Notebook",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AH7965/JupyterCellDescriberChatGPT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points = {
    },
    python_requires='>=3.6',
)