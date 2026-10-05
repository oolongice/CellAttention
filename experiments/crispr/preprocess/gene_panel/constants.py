"""Curated, pre-registered gene sets. No perturbation-derived genes occur here."""

ARTICLE_BENCHMARK_GENE_SETS = {
"direct_communication": "Spp1 Cd44 Icam1 Itgal Itgb2 Cxcl9 Cxcl10 Cxcr3 Il12a Il12b Il12rb1 Il12rb2 Cxcl12 Cxcr4".split(),
"cd44_tcell_program": "Ifng Tnf Oas3 Sell Ccr7 Tcf7 Itgae Nt5e Cd7 Ltb Havcr2 Pdcd1 Lag3 Tigit Tox Gzmb Gzmk Prf1 Nkg7 Ccl3 Ccl4 Ccl5 Il7r Ptprcap Bhlhe40 Jun".split(),
"macrophage_program": "Cd68 Adgre1 Csf1r Lyz2 C1qa C1qb C1qc Apoe Trem2 Tyrobp Cd163 Mrc1 Arg1 Nos2 Il1b Tnf Stat1 Irf1 Irf7 Socs1 Socs3".split(),
"other_article_candidates": "Lsr Cxcl14 Cd276 B2m H2-K1 H2-T23 Ifngr1 Ifngr2 H2-Eb2".split(),
}
SOURCE_GENE_SETS = {
"secreted_cytokines_chemokines": "Cxcl9 Cxcl10 Cxcl12 Cxcl14 Ccl2 Ccl3 Ccl4 Ccl5 Ccl7 Ccl8 Il1a Il1b Il6 Il10 Il12a Il12b Il15 Il18 Tnf Tgfb1 Ifng Csf1 Csf2 Csf3".split(),
"secreted_ecm": "Spp1 Lgals3 Lgals9 Fn1 Thbs1 Thbs2 Col1a1 Col1a2 Dcn Lum Timp1 Mmp9 Apoe C3".split(),
"growth_vascular": "Vegfa Vegfb Pdgfa Pdgfb Hbegf Egf Fgf2 Igf1 Angpt1 Angpt2 Kitl".split(),
"contact_associated": "Icam1 Cd274 Cd276 Cd80 Cd86 Tnfsf4 Tnfsf9 Tnfsf10 Fasl Jag1 Jag2 Dll1 Dll4".split(),
}
TARGET_GENE_SETS = {
"ifn_response": "Stat1 Stat2 Irf1 Irf7 Irf9 Isg15 Ifit1 Ifit2 Ifit3 Oas1a Oas2 Oas3 Mx1 Gbp2 Gbp4 Cxcl9 Cxcl10 Socs1 Usp18 B2m".split(),
"tcr_cytotoxicity": "Cd69 Nr4a1 Nr4a2 Fos Jun Junb Ifng Tnf Prf1 Gzmb Gzmk Nkg7 Ccl3 Ccl4 Ccl5".split(),
"exhaustion": "Pdcd1 Havcr2 Lag3 Tigit Ctla4 Tox Tox2 Entpd1 Batf Nr4a1 Nr4a2 Nr4a3".split(),
"memory_stemness": "Tcf7 Lef1 Sell Ccr7 Il7r Bcl2 Slamf6 Id3 Klf2 Itgae Nt5e Ltb".split(),
"migration": "Cxcr3 Cxcr4 Ccr2 Ccr5 Ccr7 Itgal Itgb2 S1pr1 S1pr5 Rac1 Rhoa Marcks Coro1a".split(),
"macrophage_response": "Il1b Tnf Nos2 Cxcl9 Cxcl10 Spp1 Cd163 Mrc1 Arg1 Trem2 Apoe Il12a Il12b Stat1 Stat3 Stat6 Socs1 Socs3".split(),
"stress_ros": "Hif1a Nfe2l2 Sod1 Sod2 Gpx1 Gpx4 Hmox1 Nqo1 Atf3 Atf4 Ddit3 Bax Bcl2l11 Pink1 Prkn".split(),
}
FALLBACK_MARKERS = {
"T_NK": "Ptprc Cd3d Cd3e Cd3g Trac Trbc1 Trbc2 Cd8a Cd8b1 Cd4 Nkg7 Klrd1 Klrk1 Prf1 Gzmb Gzmk Ccl5 Il7r Tcf7 Ltb Ctla4 Foxp3".split(),
"Macrophage_Monocyte": "Lyz2 Csf1r Adgre1 C1qa C1qb C1qc Tyrobp Fcerg Ctss Aif1 Apoe Trem2 Ms4a7 Lgals3 Spp1 Cd163 Mrc1 Ly6c2 Ccr2 Plac8 Cxcl9 Cxcl10".split(),
"Dendritic": "Itgax Flt3 Zbtb46 Clec10a Clec9a Xcr1 Cd74 H2-Ab1 Ccr7 Fscn1".split(),
"B_cell": "Cd79a Cd79b Ms4a1 Cd37 Cd74 H2-Aa Cd22 Cd19 Bank1 Cd83".split(),
"Neutrophil": "S100a8 S100a9 Ly6g Csf3r Retnlg Camp Ngp Mmp8 Mmp9 Lcn2".split(),
"Fibroblast": "Col1a1 Col1a2 Col3a1 Dcn Lum Col6a1 Pdgfra Pdgfrb Cxcl12 Cxcl14 Fap Acta2".split(),
"Endothelial_Perivascular": "Pecam1 Cdh5 Kdr Emcn Esam Eng Vwf Rbp7 Rgs5 Cspg4".split(),
"Tumor_Epithelial_Lung": "Epcam Krt8 Krt18 Krt19 Krt14 Krt17 Mki67 Top2a Sftpa1 Sftpa2 Sftpc Ager Hopx Scgb1a1".split(),
}

IMMUNE_PATTERNS = ("T", "NK", "B", "Macroph", "Mono", "DC", "Dendritic", "Neutroph", "MAST", "Plasma")

def inverted_categories(gene_sets):
    out = {}
    for category, genes in gene_sets.items():
        for gene in genes:
            out.setdefault(gene, []).append(category)
    return out

def source_mode(gene):
    if gene in SOURCE_GENE_SETS["contact_associated"]:
        return "contact_associated"
    if gene in SOURCE_GENE_SETS["secreted_cytokines_chemokines"] or gene in SOURCE_GENE_SETS["growth_vascular"]:
        return "diffusible"
    if gene in SOURCE_GENE_SETS["secreted_ecm"]:
        return "secreted_or_matrix_bound"
    return "uncertain"
