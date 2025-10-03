from setuptools import setup, find_packages

setup(
    name="style-guide-cleaner",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "python-docx>=0.8.11",
        "openai>=1.3.0",
        "python-dotenv>=1.0.0",
        "pandas>=2.1.0",
        "markdown>=3.5.0",
        "Pillow>=10.0.0",
        "lxml>=4.9.3",
    ],
    python_requires=">=3.8",
)
