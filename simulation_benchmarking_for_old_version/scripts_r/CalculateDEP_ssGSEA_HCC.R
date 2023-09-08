# 2020/9/23
# CHANG Cheng and DONG Qian
# Function: # (1) calculate DEP between subtypes (2) ssGSEA for each center (3) biomarker panel for each subtypes
# v1: calculate DEP using S1/S2 and S1/S3
# v2: calculate DEP using S1/non-S1

library("limma")
library("plyr")
library("dplyr")
library("RSQLite")
library("S4Vectors")
library("GSVA")
library("ComplexHeatmap")
library("circlize")

#calculate NA number
f_NA_sum <- function(x) {
  sum(is.na(x))
}
#calculate NA percentage
f_NA_percentage <- function(x) {
  mean(is.na(x))
}
# split path
split_path <- function(x) {
  if (dirname(x)==x) x else c(basename(x),split_path(dirname(x)))
}

f_ssGSEA <- function(label, pro_matrix, output_dir)
{
  # 1 DEP calculation
  output_file <- head(split_path(output_dir), n=1)
  if (grepl("Gao", output_file)) {
    row_name_type <- "gene.symbol"
    logFC_cutoff <- 0.58
  }  else {
    row_name_type <- "uniprot.accession"
    logFC_cutoff <- 1
  }
  pval_cutoff <- 0.05
  min_expression_cutoff <- 0.75
  
  # sorting
  label <- label[order(label$Subtype), ]
  data <- pro_matrix[, label$ID]
  data_log <- log2(data)
  
  # DEP calculation
  x <- table(label$Subtype)
  colnumber <- ncol(data_log)
  SI <- data_log[, 1:x[[1]]]
  SII <- data_log[, (x[[1]] + 1):(x[[1]] + x[[2]])]
  SIII <- data_log[, (x[[1]] + x[[2]] + 1):colnumber]

  numberna_SI <- apply(SI, 1, f_NA_sum)
  numberna_SII <- apply(SII, 1, f_NA_sum)
  numberna_SIII <- apply(SIII, 1, f_NA_sum)
  
  pentnumberna_SI <- apply(SI, 1, f_NA_percentage)
  Expression_SI <- 1 - pentnumberna_SI
  pentnumberna_SII <- apply(SII, 1, f_NA_percentage)
  Expression_SII <- 1 - pentnumberna_SII
  pentnumberna_SIII <- apply(SIII, 1, f_NA_percentage)
  Expression_SIII <- 1 - pentnumberna_SIII

  Expression_Frequence <-
    cbind(Expression_SIII, Expression_SII, Expression_SI)
  MAX_Expression_Frequence <- apply(Expression_Frequence, 1, max)

  data_log[is.na(data_log)] <- min(data_log, na.rm = T)

  subtype <- factor(label$Subtype)
  design <- model.matrix( ~ 0 + subtype)
  rownames(design) <- colnames(data_log)
  fit <- lmFit(data_log, design)

  cont.matrix <- makeContrasts('subtypeS2-subtypeS1', levels = design)
  fit21 <- contrasts.fit(fit, cont.matrix)
  fit21 <- eBayes(fit21, trend = TRUE)
  output_S2_S1 <- topTable(fit21,
                           coef = NULL,
                           sort.by = "none",
                           n = Inf)
  
  cont.matrix <- makeContrasts('subtypeS3-subtypeS1', levels = design)
  fit31 <- contrasts.fit(fit, cont.matrix)
  fit31 <- eBayes(fit31, trend = TRUE)
  output_S3_S1 <- topTable(fit31,
                           coef = NULL,
                           sort.by = "none",
                           n = Inf)
  
  cont.matrix <- makeContrasts('subtypeS3-subtypeS2', levels = design)
  fit32 <- contrasts.fit(fit, cont.matrix)
  fit32 <- eBayes(fit32, trend = TRUE)
  output_S3_S2 <- topTable(fit32,
                           coef = NULL,
                           sort.by = "none",
                           n = Inf)

  output <-
    cbind(output_S2_S1[, c("logFC", "adj.P.Val")], output_S3_S1[, c("logFC", "adj.P.Val")], output_S3_S2[, c("logFC", "adj.P.Val")])
  names(output) <-
    c(
      "S21.logFC",
      "S21.adj.P.Val",
      "S31.logFC",
      "S31.adj.P.Val",
      "S32.logFC",
      "S32.adj.P.Val"
    )

  gene_list <- read.delim(file.path(data_path, "sp_re_Human_140916_iRT.out"), header=TRUE)
  if (grepl("Gao", output_file)){
    output$gene.symbol <- rownames(output)
    output <- cbind(output, MAX_Expression_Frequence)
    allpro <- merge(output,
      gene_list,
      by=row_name_type,
      all=FALSE,
      sort=FALSE
    )
    allpro <- allpro[!is.na(allpro$gene.symbol), ,drop=FALSE] #remove the proteins without gene symbols
  }else{
    output$uniprot.accession <- rownames(output)
    output <- cbind(output, MAX_Expression_Frequence)
    allpro <- merge(output,
      gene_list,
      by=row_name_type,
      all.x=TRUE,
      sort=FALSE
    ) 
    allpro <- allpro[!is.na(allpro$gene.symbol), ] #remove the proteins without gene symbols
  }

  DEP_S1_up <-
    allpro[allpro$S21.logFC < 0 - logFC_cutoff &
             allpro$S31.logFC < 0 - logFC_cutoff &
             allpro$S21.adj.P.Val < pval_cutoff &
             allpro$S31.adj.P.Val < pval_cutoff &
             allpro$MAX_Expression_Frequence > min_expression_cutoff, ]
  DEP_S1_down <-
    allpro[allpro$S21.logFC > logFC_cutoff &
             allpro$S31.logFC > logFC_cutoff &
             allpro$S21.adj.P.Val < pval_cutoff &
             allpro$S31.adj.P.Val < pval_cutoff &
             allpro$MAX_Expression_Frequence > min_expression_cutoff, ]
  DEP_S2_up <-
    allpro[allpro$S21.logFC > logFC_cutoff &
             allpro$S32.logFC < 0 - logFC_cutoff &
             allpro$S21.adj.P.Val < pval_cutoff &
             allpro$S32.adj.P.Val < pval_cutoff &
             allpro$MAX_Expression_Frequence > min_expression_cutoff, ]
  
  DEP_S2_down <-
    allpro[allpro$S21.logFC < 0 - logFC_cutoff &
             allpro$S32.logFC > logFC_cutoff &
             allpro$S21.adj.P.Val < pval_cutoff &
             allpro$S32.adj.P.Val < pval_cutoff &
             allpro$MAX_Expression_Frequence > min_expression_cutoff, ]
  
  DEP_S3_up <-
    allpro[allpro$S31.logFC > logFC_cutoff &
             allpro$S32.logFC > logFC_cutoff &
             allpro$S31.adj.P.Val < pval_cutoff &
             allpro$S32.adj.P.Val < pval_cutoff &
             allpro$MAX_Expression_Frequence > min_expression_cutoff, ]
  DEP_S3_down <-
    allpro[allpro$S31.logFC < 0 - logFC_cutoff &
             allpro$S32.logFC < 0 - logFC_cutoff &
             allpro$S31.adj.P.Val < pval_cutoff &
             allpro$S32.adj.P.Val < pval_cutoff &
             allpro$MAX_Expression_Frequence > min_expression_cutoff, ]
  
  DEP_list <-
    rbind(DEP_S1_up[, c("uniprot.accession", "gene.symbol")],
          DEP_S1_down[, c("uniprot.accession", "gene.symbol")],
          DEP_S2_up[, c("uniprot.accession", "gene.symbol")],
          DEP_S2_down[, c("uniprot.accession", "gene.symbol")],
          DEP_S3_up[, c("uniprot.accession", "gene.symbol")],
          DEP_S3_down[, c("uniprot.accession", "gene.symbol")])
  DEP_list <- DEP_list[!duplicated(DEP_list$gene.symbol), ]
  DEP_list <- DEP_list[!is.na(DEP_list$gene.symbol), ]
  if (grepl("Gao", output_file)){
    data_log$gene.symbol <- rownames(data_log)
  } else{
    data_log$uniprot.accession <- rownames(data_log)
  }
  GSEA_input <-
    merge(DEP_list,
          data_log,
          by = row_name_type,
          all = FALSE,
          sort = FALSE)
  GSEA_input <- GSEA_input[GSEA_input$gene.symbol != "", ]
  rownames(GSEA_input) <- GSEA_input$gene.symbol
  GSEA_input <- GSEA_input[, -c(1:2)]
  GSEA_input_mat <- as.matrix(GSEA_input)
  mat_dim <- dim(GSEA_input_mat)
  gene_num <- mat_dim[1]
  print(gene_num)
  # 2 ssGSEA
  gs <-
    read.csv(
      file.path(data_path, "geneset.csv"),
      stringsAsFactors = FALSE,
      check.names = FALSE,
      header = TRUE
    )
  gs <- as.list(gs)
  gs <- na.omit(gs)

  ssgsea_score <-
    gsva(
      GSEA_input_mat,
      gs,
      method = "ssgsea",
      min.sz = 10,
      ssgsea.norm = TRUE,
      verbose = TRUE
    )
  ssgsea_score <- as.data.frame(ssgsea_score)
  ssgsea_score$ID <- rownames(ssgsea_score)
  # write.csv(ssgsea_score, file = paste0(output_dir, "_score.csv"))

  # 3 calculate differentially expressed terms
  selected_terms <-
    read.csv(file.path(data_path, "Nature_Extended_Fig7a_terms_Neutrophils_Immune.csv"), header=TRUE)
  
  # 2020/11/17 plot heatmap for each subtype (Nature Extended Data Fig. 6a) ####
  ssgsea_score_select <-
    merge(
      selected_terms,
      ssgsea_score,
      by = "ID",
      all = FALSE,
      sort = FALSE
    )
  
  rownames(ssgsea_score_select) <- ssgsea_score_select$ID
  ssgsea_score_select <- ssgsea_score_select[,-c(1, 3)]
  
  ssgsea_score_mean2 <-
    aggregate(. ~ name, data = ssgsea_score_select, mean)
  
  rownames(ssgsea_score_mean2) <- ssgsea_score_mean2$name
  ssgsea_score_mean2 <- ssgsea_score_mean2[,-1]
  ssgsea_score_mean2_t <- as.data.frame(t(ssgsea_score_mean2))
  ssgsea_score_mean2_t$ID <- rownames(ssgsea_score_mean2_t)
  
  all_mean2 <-
    merge(
      ssgsea_score_mean2_t,
      label,
      by = "ID",
      all = FALSE,
      sort = FALSE
    )
  all_mean2$Subtype <-
    factor(all_mean2$Subtype,
           order = TRUE,
           levels = c("S1", "S2", "S3"))
  rownames(all_mean2) <- all_mean2$ID
  all_mean2 <- all_mean2[, -1]
  
  subtype_mean <-
    aggregate(. ~ Subtype, data = all_mean2, mean)
  rownames(subtype_mean) <- subtype_mean$Subtype
  subtype_mean <- subtype_mean[, -1]
  subtype_mean_t <- as.data.frame(t(subtype_mean))

  # z-score for row
  subtype_mean_t <-
    as.data.frame(t(scale(
      t(subtype_mean_t),
      center = TRUE,
      scale = TRUE
    )), stringsAsFactors = FALSE)
  
  write.csv(subtype_mean_t, file = paste0(output_dir, "_subtype_score.csv"))
  
  ann_terms<-selected_terms[, c(2, 3)]
  ann_terms <- ann_terms[!duplicated(ann_terms$name), ]
  subtype_mean_t$name<-rownames(subtype_mean_t)
  subtype_mean_t <-
    merge(
      subtype_mean_t,
      ann_terms,
      by = "name",
      all = F,
      sort = F
    )
  
  subtype_mean_t$type<-factor(subtype_mean_t$type, levels = c("Metabolism", "Proliferation", "Immune", "Metastasis", "Signaling"), labels = c("Metabolism", "Proliferation", "Immune", "Metastasis", "Signaling"))
  subtype_mean_t<-subtype_mean_t[order(subtype_mean_t$type),]
  
  h3 <-
    Heatmap(
      as.matrix(subtype_mean_t$type),
      show_row_names = F,
      show_column_names = F,
      col = c(
        Metabolism = "orange2",
        Proliferation = "royalblue",
        Immune = "yellow",
        Metastasis = "skyblue",
        Signaling = "brown1"
      ),
      heatmap_legend_param = list(title = "type")
    )
  
  ann2 <- as.data.frame(rownames(subtype_mean))
  names(ann2) <- "Subtype"
  rownames(ann2) <- ann2$Subtype
  
  column_ha <- HeatmapAnnotation(df = ann2, col = list(Subtype = c(
    S1 = "dodgerblue3",S2 = "orange2", S3 = "indianred2"
  )))
  colno <- colorRamp2(c(-2, 0, 2), c("#4989BE", "white", "#EE3B3B"))
  
  rownames(subtype_mean_t)<-subtype_mean_t$name
  input<-subset(subtype_mean_t, select = -c(type,name))
  
  h2<-Heatmap(as.matrix(input),
              cluster_rows =F,
              cluster_columns = F,
              col = colno,
              top_annotation = column_ha,
              column_title = " ", 
              column_split=colnames(input),
              rect_gp = gpar(col= "white",lwd=3),
              column_names_gp = gpar(fontsize = 2),
              row_names_gp = gpar(fontsize =9),
              row_gap = 2,
              show_heatmap_legend = T,
              heatmap_legend_param = list(
                title= "enrichment score", at = c(-2,0,2), 
                title_gp = gpar(col = "black"),
                labels = c(-2,0,2),
                title_position = "leftcenter-rot", 
                legend_height=unit(5,"cm"), legend_direction="vertical"),
              show_row_names = T,show_column_names = F)
  
  png(paste0(output_dir,'_heatmap.png'),width=1500,height=1200,res=200)
  h <- h3 + h2
  print(h)
  dev.off()
  
  subtype_mean_t
}

f_fake_GSEA_result <- function(save_dir){
  S1 <- c(0, 0, 0, 0, 0)
  S2 <- c(0, 0, 0, 0, 0)
  S3 <- c(0, 0, 0, 0, 0)
  df <- data.frame(S1, S2, S3)
  write.csv(df, file = paste0(save_dir, "_subtype_score.csv"))
}

# set params
root_path <- file.path("data")
data_path <- file.path(root_path, "r_data")
fold_num <- 5
dataset_names <- c("Jiang", "Gao")

# load raw data
load(file.path(data_path, "HCC_raw_data.RData"))

pro_matrix_list <- list()
for (dataset_id in 1:length(dataset_names)){
  pro_matrix_list[[dataset_id]] <- eval(as.name(paste0(dataset_names[dataset_id], "_data")))
}

# find all the result directories
dirs <- c()
for (fold in 1:fold_num){
  dirs_fold <- list.dirs(
    path=file.path(root_path, "Jiang2Gao", paste0("seed", fold-1)), 
    full.names = TRUE, 
    recursive = TRUE
  )
  dirs <- append(dirs, dirs_fold)
}

dir_lens <- c()
for (i in 1:length(dirs)){
  dir_lens[i] <- strtoi(length(split_path(dirs[i])))
}
leaf_dirs <- dirs[dir_lens==max(dir_lens)]

# loop through all the leaf directories
for (leaf_dir in leaf_dirs){
  start_time <- Sys.time()

  # load assignments
  fold_results <- list()
  print(leaf_dir)
  for (fold in 1:fold_num){
    fold_result <- read.csv(
      file.path(leaf_dir, paste0("fold-", fold - 1, ".csv")),
      sep = ",",
      header = TRUE,
      encoding = "UTF-8"
    )
    fold_results[[fold]] <- fold_result
  }
  
  # combine the result of all folds
  results <- do.call(rbind, fold_results)

  for (dataset_id in 1:length(dataset_names)){
    # allocate results to each dataset
    if (dataset_id == 1){
      labels <- results[results[, "cohort"]==(dataset_id - 1), c("patients", "label")]
    } else{
      labels <- results[results[, "cohort"]==(dataset_id - 1), c("patients", "assignment")]
    }

    colnames(labels) <- c("ID","Subtype")
    labels$Subtype <- mapvalues(labels$Subtype, 
                                from=c(0,1,2), 
                                to=c("S1","S2","S3"))

    # label <- labels
    # pro_matrix <- pro_matrix_list[[dataset_id]]
    # output_dir <- file.path(leaf_dir, paste0("GSEA_", dataset_names[dataset_id]))
    res <- try(
      f_ssGSEA(
        label=labels, 
        pro_matrix=pro_matrix_list[[dataset_id]], 
        output_dir=file.path(leaf_dir, paste0("GSEA_", dataset_names[dataset_id]))
      )
    )
    if ('try-error' %in% class(res)) {
      f_fake_GSEA_result(save_dir=file.path(leaf_dir, paste0("GSEA_", dataset_names[dataset_id])))
    } 
  }
  
  end_time <- Sys.time()
  print(end_time - start_time)
} 




