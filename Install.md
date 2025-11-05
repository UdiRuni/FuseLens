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
```
curl -O http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
```
Tools to be used: 

[Samtools](https://www.htslib.org/)

[BWA](https://bio-bwa.sourceforge.net/)

Create a new conda environment:

```python
_conda create -c conda-forge -c bioconda -n read_alignment samtools awscli bwa -y
conda activate read_alignment_
```
To perform sequencing alignment for the data we have first start by creating the index files for the bwa mem algorthim that we will use for mapping and alignment. 
Using the below command on the hg38.fa genome:
```python
bwa index hg38.fa
```
Or download the index files from a publicly open AWS s3 bucket:

This is the hg38.fa (no need to re-download if you already have it, just rename)
```python
aws s3 --no-sign-request --region eu-west-1 cp s3://ngi-igenomes/igenomes/Homo_sapiens/UCSC/hg38/Sequence/BWAIndex/genome.fa .
```
These are the index files
```python
aws s3 --no-sign-request --region eu-west-1 cp s3://ngi-igenomes/igenomes/Homo_sapiens/UCSC/hg38/Sequence/BWAIndex/genome.fa.amb .
aws s3 --no-sign-request --region eu-west-1 cp s3://ngi-igenomes/igenomes/Homo_sapiens/UCSC/hg38/Sequence/BWAIndex/genome.fa.ann .
aws s3 --no-sign-request --region eu-west-1 cp s3://ngi-igenomes/igenomes/Homo_sapiens/UCSC/hg38/Sequence/BWAIndex/genome.fa.bwt .
aws s3 --no-sign-request --region eu-west-1 cp s3://ngi-igenomes/igenomes/Homo_sapiens/UCSC/hg38/Sequence/BWAIndex/genome.fa.pac .
aws s3 --no-sign-request --region eu-west-1 cp s3://ngi-igenomes/igenomes/Homo_sapiens/UCSC/hg38/Sequence/BWAIndex/genome.fa.sa .
```
The below command will align the RNA Seq samples given all files are present in the current working directory:

```python
samtools index -@ 8 AD0699_18N.bam
```
