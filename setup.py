from setuptools import setup

setup(
    name="maverick-coref",
    version="1.0.3",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sapienzanlp/maverick-coref",
    packages=["maverick", "maverick.models", "maverick.data", "maverick.utils", "maverick.common"],
    install_requires=[
        "huggingface-hub>=0.19.0",
        "hydra-core>=1.2",
        "jsonlines==4.0.0",
        "nltk==3.8.1",
        "pandas>=1.3.5",
        "protobuf==3.20",
        "pytorch-lightning>=1.8.0",
        "sentencepiece==0.2.0",
        "scipy>=1.10",
        "spacy>=3.7.2",
        "transformers>=4.34",
    ],
    python_requires=">=3.8.0",
    author="Giuliano Martinelli",
    author_email="giuliano.martinelli97@gmail.com",
    zip_safe=False,
)
