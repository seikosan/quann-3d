# METRIKLER_2026-11-20 — ham araç çıktısından üretilen sayı künyesi

Bu tablo elle yazılmadı; `uretim_metrik.py` aynı dizindeki ham JSON kayıtlarını okuyarak üretti.
Her satır: `kayıt dosyası | alan | değer`. Kayıt dosyaları bu dizindedir.

| Kayıt | Alan | Değer |
|---|---|---|
| girdi `ABSTRACT_FILING-COPY_EN.txt` | bayt / sha256(16) | 864 · 4157b9f924619f9e |
| girdi `CLAIMS_FILING-COPY_EN.txt` | bayt / sha256(16) | 10988 · 38ce1f92180f8507 |
| girdi `DESCRIPTION_FILING-COPY_EN.txt` | bayt / sha256(16) | 36677 · a228276a14fc6028 |
| girdi `ISTEMLER_FILING-COPY_TR.txt` | bayt / sha256(16) | 11715 · 50160803c4f77624 |
| girdi `OZET_FILING-COPY_TR.txt` | bayt / sha256(16) | 828 · 1ce74faf76668ad5 |
| girdi `TARIFNAME_FILING-COPY_TR.txt` | bayt / sha256(16) | 37279 · fff62fd35037fa5f |

| `A_us_claims_en_default.json` | review_patent_claims · istem (`claim_count`) | 15 |
| `A_us_claims_en_default.json` | review_patent_claims · bağımsız (`independent_claims`) | 3 |
| `A_us_claims_en_default.json` | review_patent_claims · bağımlı (`dependent_claims`) | 12 |
| `A_us_claims_en_default.json` | review_patent_claims · compliance_score (`compliance_score`) | 100 |
| `A_us_claims_en_default.json` | review_patent_claims · toplam bulgu (`total_issues`) | 0 |
| `A_us_claims_en_default.json` | review_patent_claims · kritik (`critical_issues`) | 0 |
| `A_us_claims_en_default.json` | review_patent_claims · önemli (`important_issues`) | 0 |
| `A_us_claims_en_default.json` | review_patent_claims · minör (`minor_issues`) | 0 |

| `B_us_claims_en_antecedentON.json` | ClaimsAnalyzer (opt-in antecedent ON) · istem (`claim_count`) | 15 |
| `B_us_claims_en_antecedentON.json` | ClaimsAnalyzer (opt-in antecedent ON) · bağımsız (`independent_count`) | 3 |
| `B_us_claims_en_antecedentON.json` | ClaimsAnalyzer (opt-in antecedent ON) · bağımlı (`dependent_count`) | 12 |
| `B_us_claims_en_antecedentON.json` | ClaimsAnalyzer (opt-in antecedent ON) · compliance_score (`compliance_score`) | 85.67 |
| `B_us_claims_en_antecedentON.json` | ClaimsAnalyzer (opt-in antecedent ON) · toplam bulgu (`total_issues`) | 43 |
| `B_us_claims_en_antecedentON.json` | ClaimsAnalyzer (opt-in antecedent ON) · kritik (`critical_issues`) | 0 |
| `B_us_claims_en_antecedentON.json` | ClaimsAnalyzer (opt-in antecedent ON) · önemli (`important_issues`) | 43 |
| `B_us_claims_en_antecedentON.json` | ClaimsAnalyzer (opt-in antecedent ON) · minör (`minor_issues`) | 0 |

| `C_us_spec_support_en.json` | review_specification · tarifname paragrafı (`specification_paragraphs`) | 70 |
| `C_us_spec_support_en.json` | review_specification · indekslenen terim (`indexed_terms`) | 1269 |
| `C_us_spec_support_en.json` | review_specification · toplam bulgu (`total_issues`) | 47 |
| `C_us_spec_support_en.json` | review_specification · kritik (`critical_issues`) | 0 |
| `C_us_spec_support_en.json` | review_specification · önemli (`important_issues`) | 47 |
| `C_us_spec_support_en.json` | review_specification · yazılı-tanım (`written_description_issues`) | 47 |
| `C_us_spec_support_en.json` | review_specification · enablement (`enablement_issues`) | 0 |
| `C_us_spec_support_en.json` | review_specification · compliant alanı (`compliant`) | True |

| `D_us_formal_abstract_filingcopy.json` | check_formalities · kritik (`critical_issues`) | 2 |
| `D_us_formal_abstract_filingcopy.json` | check_formalities · uyarı (`warnings`) | 0 |
| `D_us_formal_abstract_filingcopy.json` | check_formalities · bilgi (`info`) | 0 |
| `D_us_formal_abstract_filingcopy.json` | check_formalities · overall_compliant (`overall_compliant`) | False |
| `D_us_formal_abstract_filingcopy.json` | check_formalities · ready_to_file (`ready_to_file`) | False |

| `E_epo_claims_en.json` | review_epo_claims · istem (`claim_count`) | 15 |
| `E_epo_claims_en.json` | review_epo_claims · bağımsız (`independent_claims`) | 3 |
| `E_epo_claims_en.json` | review_epo_claims · bağımlı (`dependent_claims`) | 12 |
| `E_epo_claims_en.json` | review_epo_claims · toplam bulgu (`total_issues`) | 1 |
| `E_epo_claims_en.json` | review_epo_claims · kritik (`critical_issues`) | 0 |
| `E_epo_claims_en.json` | review_epo_claims · önemli (`important_issues`) | 1 |

| `F_epo_spec_support_en.json` | review_epo_specification · tarifname paragrafı (`specification_paragraphs`) | 70 |
| `F_epo_spec_support_en.json` | review_epo_specification · indekslenen terim (`indexed_terms`) | 1269 |
| `F_epo_spec_support_en.json` | review_epo_specification · toplam bulgu (`total_issues`) | 47 |
| `F_epo_spec_support_en.json` | review_epo_specification · kritik (`critical_issues`) | 0 |
| `F_epo_spec_support_en.json` | review_epo_specification · önemli (`important_issues`) | 47 |
| `F_epo_spec_support_en.json` | review_epo_specification · compliant alanı (`compliant`) | True |

| `H_epo_formal_en.json` | check_epo_formalities · kritik (`critical_issues`) | 0 |
| `H_epo_formal_en.json` | check_epo_formalities · uyarı (`warnings`) | 1 |
| `H_epo_formal_en.json` | check_epo_formalities · bilgi (`info`) | 0 |
| `H_epo_formal_en.json` | check_epo_formalities · overall_compliant (`overall_compliant`) | False |
| `H_epo_formal_en.json` | check_epo_formalities · ready_to_file (`ready_to_file`) | False |

| `K_pct_formal_en.json` | check_pct_formalities · kritik (`critical_issues`) | 1 |
| `K_pct_formal_en.json` | check_pct_formalities · uyarı (`warnings`) | 0 |
| `K_pct_formal_en.json` | check_pct_formalities · bilgi (`info`) | 0 |
| `K_pct_formal_en.json` | check_pct_formalities · overall_compliant (`overall_compliant`) | False |
| `K_pct_formal_en.json` | check_pct_formalities · ready_to_file (`ready_to_file`) | False |

| `ETR_epo_claims_tr.json` | review_epo_claims (TR nüsha) · istem (`claim_count`) | 15 |
| `ETR_epo_claims_tr.json` | review_epo_claims (TR nüsha) · bağımsız (`independent_claims`) | 15 |
| `ETR_epo_claims_tr.json` | review_epo_claims (TR nüsha) · bağımlı (`dependent_claims`) | 0 |
| `ETR_epo_claims_tr.json` | review_epo_claims (TR nüsha) · toplam bulgu (`total_issues`) | 16 |
| `ETR_epo_claims_tr.json` | review_epo_claims (TR nüsha) · kritik (`critical_issues`) | 0 |
| `ETR_epo_claims_tr.json` | review_epo_claims (TR nüsha) · önemli (`important_issues`) | 16 |

| `FTR_epo_spec_support_tr.json` | review_epo_specification (TR nüsha) · tarifname paragrafı (`specification_paragraphs`) | 70 |
| `FTR_epo_spec_support_tr.json` | review_epo_specification (TR nüsha) · indekslenen terim (`indexed_terms`) | 880 |
| `FTR_epo_spec_support_tr.json` | review_epo_specification (TR nüsha) · toplam bulgu (`total_issues`) | 0 |
| `FTR_epo_spec_support_tr.json` | review_epo_specification (TR nüsha) · kritik (`critical_issues`) | 0 |
| `FTR_epo_spec_support_tr.json` | review_epo_specification (TR nüsha) · önemli (`important_issues`) | 0 |
| `FTR_epo_spec_support_tr.json` | review_epo_specification (TR nüsha) · compliant alanı (`compliant`) | True |

| `L_package_filing.json` | check_package · taranan dosya (`files_scanned`) | 6 |
| `L_package_filing.json` | check_package · toplam bulgu (`total_issues`) | 0 |
| `L_package_filing.json` | check_package · kritik (`critical_issues`) | 0 |
| `L_package_filing.json` | check_package · paket bütünlük skoru (`package_integrity_score`) | 100 |

| `D_us_formal_abstract_filingcopy.json` | abstract alt alanı | {"word_count": 137, "line_count": 1, "compliant": true, "issues": []} |
| `D_us_formal_abstract_filingcopy.json` | title alt alanı | {"character_count": 64, "word_count": 6, "compliant": true, "issues": []} |
| `D_us_formal_abstract_filingcopy.json` | drawings alt alanı | {"figures_referenced": ["1", "2", "3", "4", "5", "6", "7", "8"], "figure_count": 8, "drawings_provided": true, "compliant": true, "issues": []} |
| `D_us_formal_abstract_filingcopy.json` | sections alt alanı | {"found_sections": {"BACKGROUND": true, "SUMMARY": false, "BRIEF DESCRIPTION OF DRAWINGS": true, "DETAILED DESCRIPTION": false}, "missing_sections": ["SUMMARY", "DETAILED DESCRIPTION"], "compliant": false} |

| `H_epo_formal_en.json` | abstract alt alanı | {"word_count": 137, "compliant": true, "issues": ["Missing reference signs"], "has_reference_signs": false} |
| `H_epo_formal_en.json` | title alt alanı | {"character_count": 64, "word_count": 6, "compliant": true, "issues": []} |
| `H_epo_formal_en.json` | drawings alt alanı | {"figures_referenced": ["1", "2", "3", "4", "5", "6", "7", "8"], "figure_count": 8, "drawings_provided": true, "compliant": true, "issues": []} |
| `H_epo_formal_en.json` | sections alt alanı | {"found_sections": {"Technical field": true, "Background art": true, "Disclosure of the invention": true, "Brief description of drawings": true, "Detailed description": true}, "missing_sections": [], "compliant": true} |
| `H_epo_formal_en.json` | claims_fees alt alanı | {"total_claims_detected": 0, "compliant": false, "fee_threshold": 15, "excess_claims": 0, "fees_apply": false} |

| `K_pct_formal_en.json` | abstract alt alanı | {"word_count": 137, "compliant": true, "issues": [], "figure_designation": "not checked: verify the request form names a figure"} |
| `K_pct_formal_en.json` | title alt alanı | {"word_count": 6, "compliant": true, "issues": []} |
| `K_pct_formal_en.json` | description alt alanı | {"found_elements": {"Technical field": true, "Background art": true, "Disclosure of the invention": true, "Brief description of drawings": true, "Best mode": true}, "missing_elements": [], "compliant": true} |
| `K_pct_formal_en.json` | claims alt alanı | {"total_claims": 0, "claim_numbers": [], "compliant": false, "issues": ["No claims found"]} |
| `K_pct_formal_en.json` | terminology alt alanı | {"non_metric_units": [], "non_standard_temperatures": [], "compliant": true} |

## Atlanan kontroller (ham `checks_skipped`)

| Kayıt | Atlanan | Gerekçe (araç çıktısı) |
|---|---|---|
| `A_us_claims_en_default.json` | `antecedent_basis` | opt-in: pattern-based antecedent checking has a high false-positive rate, so it is disabled by default |
| `B_us_claims_en_antecedentON.json` | (boş liste — hiçbir kontrol atlanmadı) | — |
| `C_us_spec_support_en.json` | (çıktıda `checks_skipped` alanı yok) | — |
| `D_us_formal_abstract_filingcopy.json` | `figure_numeral_coverage` | no detailed description supplied |
| `E_epo_claims_en.json` | (çıktıda `checks_skipped` alanı yok) | — |
| `F_epo_spec_support_en.json` | (çıktıda `checks_skipped` alanı yok) | — |
| `H_epo_formal_en.json` | `claims_fees` | no numbered claims in an identifiable claims section; supply a CLAIMS heading and claim text |
| `K_pct_formal_en.json` | `physical` | physical requirements cannot be verified from text; manual inspection required |
| `K_pct_formal_en.json` | `abstract figure designation` | the figure to publish with the abstract is chosen on the request form (PCT Rules 3.3(a)(iii) and 8.2); check the request names one |
| `ETR_epo_claims_tr.json` | (çıktıda `checks_skipped` alanı yok) | — |
| `FTR_epo_spec_support_tr.json` | (çıktıda `checks_skipped` alanı yok) | — |

## Bulgu tip dağılımları (ham `issues`)

| Kayıt | Bulgu kaydı | Tip dağılımı | Güven dağılımı |
|---|---|---|---|
| `A_us_claims_en_default.json` | 0 | — | — |
| `B_us_claims_en_antecedentON.json` | 43 | `antecedent_basis`/IMPORTANT: 43 | LOW: 43 |
| `C_us_spec_support_en.json` | 47 | `written_description`/IMPORTANT: 47 | MEDIUM: 47 |
| `D_us_formal_abstract_filingcopy.json` | 2 | `specification`/CRITICAL: 2 | HIGH: 2 |
| `E_epo_claims_en.json` | 1 | `conciseness`/IMPORTANT: 1 | HIGH: 1 |
| `F_epo_spec_support_en.json` | 47 | `sufficiency`/IMPORTANT: 47 | MEDIUM: 47 |
| `H_epo_formal_en.json` | 1 | `abstract`/IMPORTANT: 1 | HIGH: 1 |
| `K_pct_formal_en.json` | 1 | `claims`/CRITICAL: 1 | HIGH: 1 |
| `ETR_epo_claims_tr.json` | 16 | `two_part_form`/IMPORTANT: 15 · `conciseness`/IMPORTANT: 1 | HIGH: 16 |
| `FTR_epo_spec_support_tr.json` | 0 | — | — |
