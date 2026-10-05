"""Feature-namespace based symbol mapping."""
import unicodedata
import pandas as pd
ALIASES={"ICAM":"Icam1","FCGR3A":"Fcgr3","GNLY":"Nkg7","GZMH":"Gzmk","FGFBP2":"Fgfbp2"}
def clean_gene(v): return unicodedata.normalize("NFKC",str(v)).replace("\u200b","").strip().split(".")[0]
def map_gene(gene,var_names,aliases=None):
 a={**ALIASES,**(aliases or {})}; o=str(gene); g=clean_gene(gene); exact=set(map(str,var_names)); low={}
 for x in exact: low.setdefault(x.lower(),[]).append(x)
 cand=[]; method="exact"
 if g in exact: cand=[g]
 elif a.get(g,a.get(g.upper())) in exact: cand=[a.get(g,a.get(g.upper()))]; method="alias"
 elif g.lower() in low: cand=low[g.lower()]; method="case_insensitive_matrix"
 status="matched" if len(cand)==1 else ("ambiguous" if cand else "absent_from_matrix")
 return {"original_gene":o,"normalized_gene":g,"matched_expression_gene":cand[0] if len(cand)==1 else "","mapping_method":method if cand else "unmapped","is_alias":method=="alias","is_ambiguous":len(cand)>1,"status":status}
def map_table(genes,var_names,source_file="",source_category=""):
 rows=[]
 for g in genes:
  r=map_gene(g,var_names); r.update(source_file=source_file,source_category=source_category); rows.append(r)
 return pd.DataFrame(rows)
