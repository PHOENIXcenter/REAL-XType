repos_http <- 'https://mirrors.e-ducation.cn/CRAN/'

install.packages("dplyr", repos=repos_http)
install.packages("RSQLite", repos=repos_http)
install.packages("plyr", repos=repos_http)
install.packages("circlize", repos=repos_http)
install.packages("openxlsx", repos=repos_http, dependencies=TRUE)

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos=repos_http)
BiocManager::install("limma")
BiocManager::install("S4Vectors")
BiocManager::install("GSVA")
BiocManager::install("ComplexHeatmap")