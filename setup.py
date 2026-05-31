"""
Autonomous Research Stack
Build and ship autonomous LLM training research systems.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="autoresearch-stack",
    version="0.7.3",
    author="Autoresearch Team",
    author_email="turin@autoresearch.io",
    description="Autonomous LLM training research stack - build and ship autonomous research systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iknowkungfubar/autoresearch-stack",
    packages=find_packages(include=["autoresearch", "autoresearch.*"]),
    py_modules=[
        "autonomous_loop", "config", "data_intelligence", "synthetic_data",
        "curriculum", "feedback", "storage", "memory", "prioritization",
        "hypothesis", "providers", "orchestrators", "sandbox", "checkpoint",
        "monitor", "report", "figures", "stats", "paper", "peer_review",
        "metaloop", "multi_agent", "distribute", "daemon", "train_any_llm",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "autoresearch=autonomous_loop:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.md"],
    },
)
