# script for converting qs files to h5 files
rm(list=ls())

args <- commandArgs(trailingOnly = TRUE)


root_path <- file.path(args[1])
dataset <- args[2]

library(here)
library(qs)
library(rhdf5)
library(stringr)
library(SingleCellExperiment)


qs2h5 <- function(dataset, root_path) {
    sce_object <- qread(file.path(root_path, dataset, str_c(dataset, ".qs")))

    # Specify the file name
    h5_file <- file.path(root_path, dataset, str_c(dataset, ".h5"))

    # Delete the file if it exists
    if (file.exists(h5_file)) {
      deleted <- file.remove(h5_file)
      if (deleted) {
        cat("File deleted:", h5_file, "\n")
      } else {
        cat("File could not be deleted:", h5_file, "\n")
      }
    } else {
      cat("File does not exist:", h5_file, "\n")
    }

    # close all open h5 files, just in case
    h5closeAll()

    # Create the HDF5 file
    h5createFile(h5_file)

    h5write(as.numeric(colData(sce_object)[, "tricyclePosition"]), file=h5_file, name="tricyclePosition")
    h5write(as.character(colData(sce_object)[, "CCStage"]), file=h5_file, name="CCStage")

    if ("cell_type" %in% names(colData(sce_object))) {
        h5write(as.character(colData(sce_object)[, "cell_type"]), file=h5_file, name="cell_type")
    }

    if ("sample" %in% names(colData(sce_object))) {
        h5write(as.character(colData(sce_object)[, "sample"]), file=h5_file, name="sample")
    }

    if ("Gene" %in% names(rowData(sce_object))) {
cat("writing gene names\n")
        h5write(as.character(rowData(sce_object)[, "Gene"]), file=h5_file, name="Gene")
    }

    # Write data to the file with different keys

    if ("matched.PCA.s" %in% reducedDimNames(sce_object)) {
        h5write(reducedDim(sce_object, "matched.PCA.s"), file = h5_file, name = "PCA30D")
    } else if ("PCA.s" %in% reducedDimNames(sce_object)){
        h5write(reducedDim(sce_object, "PCA.s"), file = h5_file, name = "PCA30D")
    } else {
        cat("No PCA found in sce object\n")
    }

    if (dataset == "pancreas") {
        h5write(reducedDim(sce_object, "PCA.s"), file = h5_file, name = "PCA30D")

}

    if ("matched.UMAP.s" %in% reducedDimNames(sce_object)) {
        h5write(reducedDim(sce_object, "matched.UMAP.s"), file = h5_file, name = "UMAP2D")
    } else if ("UMAP.s" %in% reducedDimNames(sce_object)){
        h5write(reducedDim(sce_object, "UMAP.s"), file = h5_file, name = "UMAP2D")
    } else {
        cat("No UMAP found in sce object\n")
    }

    h5write(reducedDim(sce_object, "go.pca"), file = h5_file, name = "GOPCA20D")
    h5write(reducedDim(sce_object, "tricycleEmbedding"), file = h5_file, name = "tricycleEmbedding")

    h5closeAll()
    cat("Done writing to file:", h5_file, "\n")
}


qs2h5(dataset, root_path)