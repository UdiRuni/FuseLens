Bioinformatics software and data management are facilitated mostly using the Linux
OS. You would need to make sure you have access to a UNIX based operating system, i.e., Linux, MacOS, FreeBSD, with a
strong preference to a Linux based distribution. 

Windows OS users can enable WSL functionality and review the following guides

[What is the Windows Subsystem for Linux?](https://learn.microsoft.com/en-us/windows/wsl/about)

[How to install Linux on Windows with WSL](https://learn.microsoft.com/en-us/windows/wsl/install)

[Set up a WSL development environment](https://learn.microsoft.com/en-us/windows/wsl/setup/environment)

All tools we will be using in the scope of this project are available via the Bioconda channel. 
Working with Conda for bioinformatics will allow you to easily install packages and resolve dependency conflicts. 
Please review the [BIOCONDA](https://bioconda.github.io/) channel for more information.

The human reference genome in a fasta format from the UCSC Genome Browser ( UCSC ):

curl -O http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
