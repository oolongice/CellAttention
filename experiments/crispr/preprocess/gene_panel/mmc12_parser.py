"""Flexible long/wide marker workbook parser."""
import re
import pandas as pd
GENE_NAMES={"gene","genes","marker","markers","marker_gene","marker_genes","symbol","gene_symbol"}
CELL_NAMES={"celltype","cell_type","cell type","cluster","annotation","major_celltype","subtype"}
def standardize_cell_type(v):
 s=re.sub(r"[+\s/-]+","_",str(v).strip()).strip("_"); low=s.lower()
 for p,x in [(r"macroph|^macro","Macrophage"),(r"monocyte","Monocyte"),(r"dendritic|^dc$","Dendritic"),(r"cd8|cytotoxic","CD8_T"),(r"(^|_)t($|_)|t_cell","T_cell"),(r"(^|_)b($|_)|b_cell","B_cell"),(r"neutroph","Neutrophil"),(r"fibro|stromal","Fibroblast"),(r"endothelial|^ec$","Endothelial"),(r"tumor|malignant|cancer","Tumor"),(r"epithelial","Epithelial"),(r"smooth","SmoothMuscle"),(r"^nk$","NK")]:
  if re.search(p,low): return x
 return s
def split_genes(v):
 if pd.isna(v): return []
 out=[]
 for p in re.split(r"[,;/|\n]+",str(v).replace("\u200b"," ").strip()):
  t=p.split(); out.extend(t if len(t)>1 and all(re.fullmatch(r"[A-Za-z0-9.-]+",x) for x in t) else [p.strip()])
 return [x for x in out if x]
def inspect_mmc12(path):
 out={"path":str(path),"sheets":[]}
 for s in pd.ExcelFile(path).sheet_names:
  d=pd.read_excel(path,sheet_name=s); scores=[]
  for c in d:
   n=str(c).strip().lower(); v=d[c].dropna().astype(str); gl=float(v.str.match(r"^[A-Za-z][A-Za-z0-9.-]{1,30}$").mean()) if len(v) else 0
   scores.append({"column":str(c),"gene_score":round((.65 if n in GENE_NAMES else 0)+.35*gl,3),"celltype_score":round(.65 if n in CELL_NAMES else 0,3),"source_score":.8 if any(x in n for x in ("dataset","sample","figure","source")) else 0})
  out["sheets"].append({"sheet_name":s,"n_rows":len(d),"n_columns":len(d.columns),"columns":list(map(str,d.columns)),"non_null":{str(k):int(v) for k,v in d.notna().sum().items()},"preview":d.head().fillna("").astype(str).to_dict("records"),"candidate_scores":scores})
 return out
def parse_mmc12_markers(xlsx_path,sheet=None,gene_column=None,celltype_column=None):
 report=inspect_mmc12(xlsx_path); rows=[]
 for s in ([sheet] if sheet else [x["sheet_name"] for x in report["sheets"]]):
  d=pd.read_excel(xlsx_path,sheet_name=s); low={str(c).strip().lower():c for c in d}; gc=gene_column or next((low[x] for x in GENE_NAMES if x in low),None); cc=celltype_column or next((low[x] for x in CELL_NAMES if x in low),None)
  it=((i,r[cc],r[gc]) for i,r in d.iterrows()) if gc is not None and cc is not None else ((i,c,v) for c in d for i,v in d[c].items())
  for i,ct,v in it:
   for g in split_genes(v): rows.append({"gene_raw":g,"gene_symbol":g.strip(),"cell_type_raw":str(ct),"cell_type_standardized":standardize_cell_type(ct),"sheet_name":s,"source_row":int(i)+2,"parsing_status":"parsed"})
  next(x for x in report["sheets"] if x["sheet_name"]==s)["selected_mode"]="long" if gc is not None and cc is not None else "wide"
 t=pd.DataFrame(rows).drop_duplicates(); return t,{k:list(dict.fromkeys(x.gene_symbol)) for k,x in t.groupby("cell_type_standardized",sort=True)},report
