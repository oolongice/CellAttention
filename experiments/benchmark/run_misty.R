#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(mistyR); library(jsonlite)})
args <- commandArgs(trailingOnly=TRUE)
if (length(args)<2) stop("usage: run_misty.R MANIFEST OUTPUT_SCORES [global|receiver]")
m <- fromJSON(args[[1]]); out <- normalizePath(dirname(args[[2]]),mustWork=FALSE)
mode <- ifelse(length(args)>=3,args[[3]],"global"); dir.create(out,recursive=TRUE,showWarnings=FALSE)
genes <- scan(m$gene_ids_txt,what=character(),quiet=TRUE)
expr <- read.csv(m$expression_csv,header=FALSE,check.names=FALSE); colnames(expr)<-genes
pos <- read.csv(m$coordinates_csv,header=FALSE); colnames(pos)<-c("x","y")
groups <- scan(m$cell_groups_txt,what=integer(),quiet=TRUE)
pairs <- read.csv(m$candidate_pairs_csv,stringsAsFactors=FALSE)
sources <- unique(pairs$source_gene); targets <- unique(pairs$target_gene)

# Build source paraview once on all cells so receiver-specific fitting still sees senders.
source.views <- create_initial_view(as.data.frame(expr[,sources,drop=FALSE]))
source.views <- add_paraview(source.views,pos,l=130,family="exponential")
para.name <- setdiff(names(source.views),c("intraview","misty.uniqueid"))[[1]]
source.para <- source.views[[para.name]]$data

extract_scores <- function(result.path, group=NULL) {
  imp <- collect_results(result.path)$importances
  if (is.list(imp) && !is.data.frame(imp)) imp <- do.call(rbind,imp)
  nl <- tolower(names(imp))
  pick <- function(options) {
    hit <- which(nl %in% options)
    if (!length(hit)) stop(paste("MISTy importance schema unsupported:",paste(names(imp),collapse=",")))
    names(imp)[hit[[1]]]
  }
  sc<-pick(c("predictor","source","marker")); tc<-pick(c("target","response"))
  vc<-pick(c("importance","value","mean")); wc<-pick(c("view","view.abbrev"))
  z<-imp[grepl("para.synthetic",imp[[wc]],fixed=TRUE),,drop=FALSE]
  ans<-data.frame(source_gene=as.character(z[[sc]]),target_gene=as.character(z[[tc]]),score=as.numeric(z[[vc]]))
  ans<-merge(pairs,ans,by=c("source_gene","target_gene"),all.x=TRUE)
  ans$score[is.na(ans$score)] <- 0
  if (!is.null(group)) ans$target_group<-group
  ans
}

fit_one <- function(index,result.path) {
  target.block <- expr[index,targets,drop=FALSE]
  modeled.targets <- targets[apply(target.block,2,var) > 1e-12]
  if (!length(modeled.targets)) stop("receiver group has no variable targets")
  target.views<-create_initial_view(as.data.frame(target.block[,modeled.targets,drop=FALSE]))
  views<-add_views(target.views,create_view("paraview.synthetic_sources.130",
    as.data.frame(source.para[index,,drop=FALSE]),"para.synthetic.130"))
  run_misty(views,results.folder=result.path,target.subset=modeled.targets,
    bypass.intra=TRUE,cv.folds=5,seed=168)
}

if (mode=="global") {
  path<-fit_one(seq_len(nrow(expr)),file.path(out,"raw"))
  scores<-extract_scores(path)
} else if (mode=="receiver") {
  blocks<-list()
  receiver.groups<-sort(unique(groups[groups<6]))
  for (g in receiver.groups) {
    index<-which(groups==g)
    path<-fit_one(index,file.path(out,"raw",paste0("group_",g)))
    blocks[[length(blocks)+1]]<-extract_scores(path,g)
  }
  scores<-do.call(rbind,blocks)
} else stop(paste("unknown mode:",mode))
write.csv(scores,args[[2]],row.names=FALSE,quote=FALSE)
write_json(list(method="MISTy",version=as.character(packageVersion("mistyR")),mode=mode,
  source_view="exponential paraview, l=130",receiver_fit=ifelse(mode=="receiver","separate oracle receiver-group fits","all cells"),
  cv_folds=5,edge_truth_used_during_fit=FALSE,receiver_labels_used=(mode=="receiver")),file.path(out,"method_metadata.json"),auto_unbox=TRUE,pretty=TRUE)
