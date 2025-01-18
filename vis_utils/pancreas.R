# Modified version of processEndo.R in https://github.com/hansenlab/tricycle_paper_figs.git/scripts

rm(list=ls())
library(here)

args <- commandArgs(trailingOnly = TRUE)
root_path <- file.path(args[1])
vis_utils_path  <- file.path(args[2])

source(file.path(vis_utils_path, "tricycle_paper_figs/scripts/utils.R"))


### read in sce
dataset <- "pancreas"
filepath <- file.path(root_path, dataset, str_c(dataset, ".qs"))
sce.o <- qread(filepath)


dataname <- "mPancreas"
species <- "mouse"
point.size <- 3.01
point.alpha <- 0.6
gene <- rownames(sce.o)
GENE <- toupper(gene)
ensembl <- NULL

### get go pca
GO.o <- getGO(sce.o, row.id = gene, id.type = c("SYMBOL"), species = species, ncomponents = 20, seed = 100,
							runSeuratBy = NULL)
go.pca.m <- reducedDim(GO.o, "PCA.s")
reducedDim(sce.o, "go.pca") <- go.pca.m

### extract genes
sce.o$top2a <- assay(sce.o, "log.s")[which(GENE == "TOP2A"), ]
sce.o$smc4 <- assay(sce.o, "log.s")[which(GENE == "SMC4"), ]

### modified version from https://github.com/hansenlab/tricycle_paper_figs.git/scripts/utils.R.
### Directly loads neurRef from tricycle repo instead of using installed tricycle library
getPct0 <- function(sce.o, GENE) {
load(file.path(vis_utils_path, "tricycle", "data", "neuroRef.rda"))
ref.df <- neuroRef
data.m <- assay(sce.o, "log.s")[GENE %in% ref.df$SYMBOL, ]
pct0.v <- colMeans(data.m < 0.01)
return(pct0.v)
}

### get pct0
sce.o$pct0 <- getPct0(sce.o, GENE)

### get 5 stage assignment
sce.o <- estimate_Schwabe_stage(sce.o, gname = gene, batch.v = sce.o$sample, exprs_values = "log.s", gname.type = "SYMBOL", species = species)

### get theta and our projection
sce.o <- estimate_cycle_position(sce.o, gname = gene, gname.type = "SYMBOL", species = species, exprs_values = "log.s")

### save updated .qs
qsave(qsave(sce.o, file=file.path(root_path, dataset, str_c(dataset, ".qs")))
)
